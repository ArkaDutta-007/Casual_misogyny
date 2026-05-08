"""
Persona-driven comment generator for the casual misogyny annotation study.

Each row in Misogyny_Persona.csv defines a specific character + situation.
This script asks LLMs to generate the realistic comment/message that character
would write or say in that situation.

Execution order : GPT → Gemini → Claude  (sequential across models, configurable via --models)
Parallelism     : within each model, all personas run concurrently up to --concurrency limit.
                  GPT & Claude use asyncio; Gemini (sync SDK) uses ThreadPoolExecutor.
Resumability    : per-persona atomic JSON cache (rename-swap); stale .tmp files cleaned on
                  startup; Ctrl-C safe — re-run resumes from the last completed persona.
Streaming       : per-model JSONL  outputs/persona_comments/{model}.jsonl — one record per
                  line, flushed and fsynced immediately after each API response.
Progress        : per-model tqdm progress bar — tmux-compatible (stderr, dynamic_ncols).

Usage:
    python generate_persona_comments.py [--models gpt gemini claude]
                                        [--personas P001 P002 ...]
                                        [--frames "Workplace: Menstrual..."]
                                        [--concurrency 4]
                                        [--output-dir outputs/persona_comments]

Requires:
    ANTHROPIC_API_KEY  OPENAI_API_KEY  GOOGLE_API_KEY  (in .env or environment)
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import google.generativeai as genai
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

# Load .env from the same directory as this script
load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Config (all overridable via .env)
# ---------------------------------------------------------------------------

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
GPT_MODEL    = os.getenv("GPT_MODEL",    "gpt-4o")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TEMPERATURE  = 1.0
MAX_TOKENS   = 2048

DEFAULT_MODEL_ORDER = ["gpt", "gemini", "claude"]

PERSONA_CSV = Path(__file__).parent / "Misogyny_Persona.csv"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a research assistant for an NLP + Computational Social Science study "
    "on casual misogyny in India. You are generating training data for a misogyny "
    "detection classifier. Generate realistic examples that researchers will annotate."
)

SYSTEM_PROMPT_FALLBACK = (
    "You are generating training data for a misogyny detection classifier. "
    "Generate realistic examples that researchers will annotate."
)


def build_user_prompt(persona: dict) -> str:
    pid    = persona["persona_id"]
    frame  = persona["frame"]
    ptext  = persona["persona"]

    return f"""
You are generating a realistic comment or message for a specific character in a
research study on casual misogyny in India.

--- CHARACTER & SITUATION ---
{ptext}

--- FRAME (type of social situation / potential misogynistic framing) ---
{frame}

--- TASK ---
Write the realistic comment, message, or dialogue that THIS character would
actually write or say in the situation described above.

Requirements:
- Write authentically in this character's voice, register, and social position.
- Reflect their education, class, region, caste background, and relationships.
- Use appropriate code-switching (Hindi-English mix, or regional language hints)
  if natural for this character. Do NOT force it if it's not natural.
- The character must NOT sound like they know they might be misogynistic.
  Their view should feel genuine and internally consistent.
- Survey-safe: no slurs, no graphic sexual content.
- Length appropriate to the medium implied by the situation:
    WhatsApp / text chat: 1–3 sentences, casual
    Spoken conversation : 2–4 short turns ("Speaker A: ... Speaker B: ...")
    Social media post   : 2–5 sentences, might use hashtags
    Formal / workplace  : complete sentences, professional register
- If the situation describes a reaction or comment about something, generate
  that reaction — not a description of it.

Output ONLY a valid JSON object (no markdown fences, no prose before or after):
{{
  "persona_id": "{pid}",
  "frame": "{frame}",
  "generated_comment": "...",
  "label": "misogynistic or non_misogynistic",
  "covertness": "overt or covert (omit if non_misogynistic)",
  "annotation_note": "1–2 sentences: why this is or isn't misogynistic, and which surface manifestation it encodes"
}}
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_personas(csv_path: Path) -> list[dict]:
    """Load Misogyny_Persona.csv and assign stable persona IDs (P001…Pnnn)."""
    personas = []
    with open(csv_path, encoding="latin-1", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            personas.append({
                "persona_id": f"P{i:03d}",
                "frame":      row["Frame"].strip(),
                "persona":    row["Persona"].strip(),
            })
    return personas


def extract_json(raw: str) -> dict | None:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def atomic_write(path: Path, data: dict) -> None:
    """Write JSON to a .tmp file then atomically rename — safe against mid-write crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def validate_cache(path: Path) -> dict | None:
    """Load cached persona JSON; return None (and delete) if missing or corrupt."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "generated_comment" not in data or not data["generated_comment"]:
            raise ValueError("missing generated_comment")
        return data
    except Exception as e:
        tqdm.write(f"  [cache] corrupt {path.name}: {e} — will regenerate", file=sys.stderr)
        path.unlink(missing_ok=True)
        path.with_suffix(".tmp").unlink(missing_ok=True)
        return None


def append_jsonl(path: Path, record: dict) -> None:
    """Append a record to a JSONL file and fsync."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def cleanup_tmp_files(output_dir: Path, models: list[str]) -> None:
    """Delete any stale .tmp files left by a previous interrupted run."""
    count = 0
    for model in models:
        model_dir = output_dir / model
        if model_dir.exists():
            for tmp in model_dir.glob("*.tmp"):
                tmp.unlink(missing_ok=True)
                count += 1
    if count:
        tqdm.write(f"[startup] cleaned {count} stale .tmp file(s)", file=sys.stderr)


# ---------------------------------------------------------------------------
# Async API callers
# ---------------------------------------------------------------------------

async def call_claude(persona: dict, client: anthropic.AsyncAnthropic) -> dict | None:
    prompt = build_user_prompt(persona)
    for system in (SYSTEM_PROMPT, SYSTEM_PROMPT_FALLBACK):
        try:
            async with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                final = await stream.get_final_message()
            text = next((b.text for b in final.content if b.type == "text"), "")
            result = extract_json(text)
            if result:
                return result
        except anthropic.BadRequestError:
            if system == SYSTEM_PROMPT_FALLBACK:
                return None
            continue
        except anthropic.RateLimitError as e:
            wait = int(e.response.headers.get("retry-after", "60"))
            tqdm.write(f"  [claude] rate-limited — sleeping {wait}s", file=sys.stderr)
            await asyncio.sleep(wait)
            return await call_claude(persona, client)
        except Exception as e:
            tqdm.write(f"  [claude] error: {e}", file=sys.stderr)
            return None
    return None


# Module-level flag so all coroutines share the same knowledge once discovered
_gpt_use_completion_tokens: bool = False


def _gpt_kwargs() -> dict:
    kwargs: dict = {"model": GPT_MODEL}
    if _gpt_use_completion_tokens:
        kwargs["max_completion_tokens"] = MAX_TOKENS
    else:
        kwargs["max_tokens"] = MAX_TOKENS
        kwargs["temperature"] = TEMPERATURE
    return kwargs


async def call_gpt(persona: dict, client: AsyncOpenAI) -> dict | None:
    global _gpt_use_completion_tokens
    prompt = build_user_prompt(persona)
    for system in (SYSTEM_PROMPT, SYSTEM_PROMPT_FALLBACK):
        try:
            resp = await client.chat.completions.create(
                **_gpt_kwargs(),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
            )
            text = resp.choices[0].message.content or ""
            result = extract_json(text)
            if result:
                return result
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg:
                tqdm.write("  [gpt] rate-limited — sleeping 60s", file=sys.stderr)
                await asyncio.sleep(60)
                return await call_gpt(persona, client)
            if "max_tokens" in msg and "max_completion_tokens" in msg:
                if not _gpt_use_completion_tokens:
                    tqdm.write("  [gpt] switching to max_completion_tokens (once)", file=sys.stderr)
                    _gpt_use_completion_tokens = True
                continue
            if system == SYSTEM_PROMPT:
                tqdm.write(f"  [gpt] retrying with fallback prompt: {e}", file=sys.stderr)
                continue
            tqdm.write(f"  [gpt] error: {e}", file=sys.stderr)
            return None
    return None


def _gemini_sync(persona: dict, model) -> dict | None:
    """Synchronous Gemini call — dispatched via ThreadPoolExecutor."""
    for attempt, sys_p in enumerate((SYSTEM_PROMPT, SYSTEM_PROMPT_FALLBACK)):
        prompt = sys_p + "\n\n" + build_user_prompt(persona)
        try:
            resp = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_TOKENS,
                ),
            )
            result = extract_json(resp.text or "")
            if result:
                return result
        except Exception as e:
            msg = str(e).lower()
            if "quota" in msg or "429" in msg or "resource exhausted" in msg:
                tqdm.write("  [gemini] quota/rate-limit — sleeping 60s", file=sys.stderr)
                time.sleep(60)
                return _gemini_sync(persona, model)
            tqdm.write(f"  [gemini] error (attempt {attempt + 1}): {e}", file=sys.stderr)
            if attempt == 0:
                time.sleep(3)
    return None


async def call_gemini(persona: dict, model, executor: ThreadPoolExecutor) -> dict | None:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _gemini_sync, persona, model)


# ---------------------------------------------------------------------------
# Per-persona, per-model runner  (bounded by a shared semaphore)
# ---------------------------------------------------------------------------

async def run_model_persona(
    persona: dict,
    model: str,
    clients: dict,
    output_dir: Path,
    jsonl_path: Path,
    jsonl_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor | None,
) -> tuple[str, dict | None]:
    """Process one persona for one model; return (persona_id, result_or_None)."""
    pid        = persona["persona_id"]
    cache_path = output_dir / model / f"{pid}.json"

    async with semaphore:
        # Fast path: already cached
        cached = validate_cache(cache_path)
        if cached:
            tqdm.write(f"  [{model:6}] {pid} — cache hit", file=sys.stderr)
            return pid, cached

        # Call the appropriate model
        if model == "gpt":
            raw = await call_gpt(persona, clients["gpt"])
        elif model == "gemini":
            raw = await call_gemini(persona, clients["gemini"], executor)
        elif model == "claude":
            raw = await call_claude(persona, clients["claude"])
        else:
            raw = None

        if raw is None:
            tqdm.write(f"  [{model:6}] {pid} — no result", file=sys.stderr)
            return pid, None

        # Enrich with source metadata
        result = {
            **raw,
            "persona_id":   pid,
            "frame":        persona["frame"],
            "persona_text": persona["persona"],
            "source_model": model,
        }

        # Persist
        atomic_write(cache_path, result)

        async with jsonl_lock:
            append_jsonl(jsonl_path, result)

        return pid, result


# ---------------------------------------------------------------------------
# Per-model runner across all personas  (parallel with tqdm progress bar)
# ---------------------------------------------------------------------------

async def run_model_all_personas(
    personas: list[dict],
    model: str,
    clients: dict,
    output_dir: Path,
    jsonl_path: Path,
    concurrency: int,
    executor: ThreadPoolExecutor | None,
) -> dict[str, dict]:
    semaphore  = asyncio.Semaphore(concurrency)
    jsonl_lock = asyncio.Lock()
    results: dict[str, dict] = {}

    tasks = [
        asyncio.create_task(
            run_model_persona(
                persona, model, clients,
                output_dir, jsonl_path, jsonl_lock, semaphore, executor,
            )
        )
        for persona in personas
    ]

    with tqdm(
        total=len(tasks),
        desc=f"  [{model:6}]",
        unit="persona",
        dynamic_ncols=True,
        file=sys.stderr,
        colour="cyan",
    ) as pbar:
        for fut in asyncio.as_completed(tasks):
            try:
                pid, result = await fut
            except Exception as e:
                tqdm.write(f"  [{model}] unexpected error: {e}", file=sys.stderr)
                pbar.update(1)
                continue
            if result is not None:
                pbar.set_postfix_str(f"last={pid}")
                results[pid] = result
            else:
                pbar.set_postfix_str(f"last={pid} FAILED")
            pbar.update(1)

    return results


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------

async def main_async(args) -> None:
    personas = load_personas(PERSONA_CSV)

    # Filter by persona ID if requested
    if args.personas:
        personas = [p for p in personas if p["persona_id"] in args.personas]
        if not personas:
            sys.exit(f"[ERROR] no matching persona IDs: {args.personas}")

    # Filter by frame substring if requested
    if args.frames:
        personas = [p for p in personas
                    if any(f.lower() in p["frame"].lower() for f in args.frames)]
        if not personas:
            sys.exit(f"[ERROR] no personas match frames: {args.frames}")

    models: list[str] = args.models
    output_dir = Path(args.output_dir)

    print(f"Personas    : {len(personas)}")
    print(f"Models      : {' → '.join(models)}  (sequential)")
    print(f"Concurrency : {args.concurrency} per model")
    print(f"Output dir  : {output_dir}")

    cleanup_tmp_files(output_dir, models)

    # -- Init API clients ----------------------------------------------------
    clients: dict = {}
    if "gpt" in models:
        clients["gpt"] = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if "gemini" in models:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        clients["gemini"] = genai.GenerativeModel(GEMINI_MODEL)

    if "claude" in models:
        clients["claude"] = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    all_results: dict[str, dict[str, dict]] = {}
    executor = ThreadPoolExecutor(max_workers=args.concurrency) if "gemini" in models else None

    try:
        for model in models:
            jsonl_path = output_dir / f"{model}.jsonl"
            print(f"\n{'='*60}")
            print(f"  Model: {model.upper()}")
            print(f"{'='*60}")
            results = await run_model_all_personas(
                personas, model, clients, output_dir, jsonl_path,
                args.concurrency, executor,
            )
            all_results[model] = results
            n_ok   = len(results)
            n_fail = len(personas) - n_ok
            print(f"  [{model}] done — {n_ok} ok, {n_fail} failed")
    finally:
        if executor:
            executor.shutdown(wait=False)

    # -- Merge all models into one flat JSON list ----------------------------
    merged: list[dict] = []
    seen_pids: set[str] = set()
    for model in models:
        for pid, rec in all_results.get(model, {}).items():
            merged.append(rec)

    merged_path = output_dir / "merged_persona_comments.json"
    merged_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nMerged {len(merged)} records → {merged_path}")

    # -- Per-frame summary ---------------------------------------------------
    from collections import Counter
    frame_counts = Counter(r["frame"] for r in merged)
    print("\nPer-frame counts (across all models):")
    for frame, cnt in sorted(frame_counts.items(), key=lambda x: -x[1]):
        print(f"  {cnt:4d}  {frame}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate persona-driven comments for the casual misogyny study."
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODEL_ORDER,
        choices=["gpt", "gemini", "claude"],
        help="Models to run (in order). Default: gpt gemini claude",
    )
    parser.add_argument(
        "--personas", nargs="+", metavar="PID",
        help="Restrict to specific persona IDs (e.g. P001 P002). Default: all.",
    )
    parser.add_argument(
        "--frames", nargs="+", metavar="FRAME",
        help="Restrict to personas whose frame contains this substring (case-insensitive).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Max concurrent API calls per model. Default: 5.",
    )
    parser.add_argument(
        "--output-dir", default="outputs/persona_comments",
        help="Directory for per-persona JSON cache and JSONL outputs.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
