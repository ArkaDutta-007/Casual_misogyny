"""
Frame × Persona cross-product comment generator.

Takes the 16 misogyny frames from misogyny_frames.json and the 230 persona
STYLE PROFILES from Misogyny_Persona.csv, and generates one comment per
(persona, frame) pair.

Key difference from generate_persona_comments.py
-------------------------------------------------
Each CSV persona text contains both:
  (a) identity attributes  — name, age, region, occupation, class, caste, etc.
  (b) a specific situation — "you are texting your friends after…"

Here we use ONLY the identity attributes (a) and replace the situation with a
fresh scenario drawn from the JSON frame's example_scenarios, surface_manifestations,
and india_specific_context.  The LLM is explicitly instructed to ignore the
CSV situation and instead inhabit the frame context.

Scale    : 230 personas × 16 frames = 3,680 pairs per model
Cache key: {persona_id}_{frame_id}.json  (e.g. P001_F03.json)
Output   : outputs/frame_persona_cross/{model}/
           outputs/frame_persona_cross/{model}.jsonl
           outputs/frame_persona_cross/merged_frame_persona_cross.json

Usage:
    python generate_frame_persona_cross.py [--models gpt gemini claude]
                                           [--personas P001 P002 ...]
                                           [--frames F01 F03 F07]
                                           [--concurrency 6]
                                           [--output-dir outputs/frame_persona_cross]

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

PERSONA_CSV   = Path(__file__).parent / "Misogyny_Persona.csv"
FRAMES_JSON   = Path(__file__).parent / "misogyny_frames.json"

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


def _frame_context_block(frame: dict) -> str:
    """Build a concise but rich frame context string from the JSON frame dict."""
    lines = [
        f"Frame ID    : {frame['frame_id']}",
        f"Label       : {frame['label']}",
        f"Covertness  : {frame.get('covertness', 'medium')}",
        f"Core belief : {frame['core_belief']}",
        "",
        "Surface manifestations (how this misogyny typically shows up):",
    ]
    for m in frame.get("surface_manifestations", []):
        lines.append(f"  • {m}")

    lines.append("")
    lines.append("Linguistic cues:")
    for c in frame.get("linguistic_cues", []):
        lines.append(f"  • {c}")

    if frame.get("india_specific_context"):
        lines.append("")
        lines.append(f"India-specific context: {frame['india_specific_context']}")

    if frame.get("llm_generation_prompt_hints"):
        lines.append("")
        lines.append("Generation hints:")
        for h in frame["llm_generation_prompt_hints"]:
            lines.append(f"  → {h}")

    return "\n".join(lines)


def build_cross_prompt(persona: dict, frame: dict) -> str:
    """
    Build the user prompt for a (persona, frame) cross-product pair.

    The persona's CSV text may contain a specific situation — we instruct the
    LLM to IGNORE that situation and instead use only the person's identity
    attributes (name, age, region, occupation, class, caste, language style).
    The frame provides the new situational context.
    """
    pid   = persona["persona_id"]
    fid   = frame["frame_id"]
    ptext = persona["persona"]

    frame_block = _frame_context_block(frame)

    return f"""
You are generating a realistic comment or message for a research study on casual
misogyny in India. You will be given a PERSON PROFILE and a MISOGYNY FRAME.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERSON PROFILE (from research CSV — persona_id {pid})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ptext}

IMPORTANT: The profile above may describe a specific situation. IGNORE that
specific situation entirely. Extract ONLY the person's IDENTITY ATTRIBUTES:
  • Name and gender
  • Age and life stage
  • Location / region (city, state, rural/urban)
  • Occupation, industry, seniority
  • Class, caste background, education level (if mentioned or inferable)
  • Language style and register (formal/informal, code-switching patterns)
  • Social relationships and family context (if mentioned)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MISOGYNY FRAME (frame_id {fid}) — THIS is the situation context
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{frame_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Imagine THIS PERSON (using only their identity attributes above) in a realistic
social situation related to the MISOGYNY FRAME.  Write the specific comment,
message, or short exchange they would make.

Requirements:
- Authentically match their voice: education level, region, class, caste, age.
- Use appropriate code-switching (Hindi-English, or regional language hints)
  only if natural for this person. Do NOT force it.
- The person must NOT sound like they know they are being misogynistic.
  Their view should feel genuine, internally consistent, and culturally plausible.
- Survey-safe: no slurs, no graphic sexual content.
- Pick the most natural communication medium for this person and frame:
    WhatsApp / text chat : 1–3 sentences, casual, possibly emoji
    Spoken conversation  : 2–4 short turns ("A: … B: …")
    Social media post    : 2–5 sentences, may use hashtags
    Formal / workplace   : complete sentences, professional register
- Generate the actual comment/message — NOT a description of it.

Output ONLY a valid JSON object (no markdown fences, no prose outside the JSON):
{{
  "persona_id": "{pid}",
  "frame_id":   "{fid}",
  "frame_label": "{frame['label']}",
  "generated_comment": "…",
  "label": "misogynistic or non_misogynistic",
  "covertness": "overt or covert (omit if non_misogynistic)",
  "annotation_note": "1–2 sentences: why this is or isn't misogynistic, and which surface manifestation it encodes"
}}
""".strip()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_personas(csv_path: Path) -> list[dict]:
    """Load Misogyny_Persona.csv and assign stable persona IDs (P001…Pnnn)."""
    personas = []
    with open(csv_path, encoding="latin-1", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            personas.append({
                "persona_id":       f"P{i:03d}",
                "original_frame":   row["Frame"].strip(),   # CSV frame (kept for reference)
                "persona":          row["Persona"].strip(),
            })
    return personas


def load_frames(frames_path: Path) -> list[dict]:
    """Load misogyny_frames.json; return list of frame dicts."""
    data = json.loads(frames_path.read_text(encoding="utf-8"))
    return data["frames"]


# ---------------------------------------------------------------------------
# Helpers (identical to generate_persona_comments.py)
# ---------------------------------------------------------------------------

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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def validate_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "generated_comment" not in data or not data["generated_comment"]:
            raise ValueError("missing generated_comment")
        return data
    except Exception as e:
        tqdm.write(f"  [cache] corrupt {path.name}: {e} — regenerating", file=sys.stderr)
        path.unlink(missing_ok=True)
        path.with_suffix(".tmp").unlink(missing_ok=True)
        return None


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def cleanup_tmp_files(output_dir: Path, models: list[str]) -> None:
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

async def call_claude(
    persona: dict, frame: dict, client: anthropic.AsyncAnthropic
) -> dict | None:
    prompt = build_cross_prompt(persona, frame)
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
            return await call_claude(persona, frame, client)
        except Exception as e:
            tqdm.write(f"  [claude] error: {e}", file=sys.stderr)
            return None
    return None


_gpt_use_completion_tokens: bool = False


def _gpt_kwargs() -> dict:
    kwargs: dict = {"model": GPT_MODEL}
    if _gpt_use_completion_tokens:
        kwargs["max_completion_tokens"] = MAX_TOKENS
    else:
        kwargs["max_tokens"] = MAX_TOKENS
        kwargs["temperature"] = TEMPERATURE
    return kwargs


async def call_gpt(
    persona: dict, frame: dict, client: AsyncOpenAI
) -> dict | None:
    global _gpt_use_completion_tokens
    prompt = build_cross_prompt(persona, frame)
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
                return await call_gpt(persona, frame, client)
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


def _gemini_sync(persona: dict, frame: dict, model) -> dict | None:
    for attempt, sys_p in enumerate((SYSTEM_PROMPT, SYSTEM_PROMPT_FALLBACK)):
        prompt = sys_p + "\n\n" + build_cross_prompt(persona, frame)
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
                return _gemini_sync(persona, frame, model)
            tqdm.write(f"  [gemini] error (attempt {attempt + 1}): {e}", file=sys.stderr)
            if attempt == 0:
                time.sleep(3)
    return None


async def call_gemini(
    persona: dict, frame: dict, model, executor: ThreadPoolExecutor
) -> dict | None:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _gemini_sync, persona, frame, model)


# ---------------------------------------------------------------------------
# Per-(persona, frame) runner
# ---------------------------------------------------------------------------

async def run_model_pair(
    persona: dict,
    frame: dict,
    model: str,
    clients: dict,
    output_dir: Path,
    jsonl_path: Path,
    jsonl_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor | None,
) -> tuple[str, dict | None]:
    """Process one (persona, frame) pair for one model."""
    pid        = persona["persona_id"]
    fid        = frame["frame_id"]
    pair_key   = f"{pid}_{fid}"
    cache_path = output_dir / model / f"{pair_key}.json"

    async with semaphore:
        cached = validate_cache(cache_path)
        if cached:
            return pair_key, cached

        if model == "gpt":
            raw = await call_gpt(persona, frame, clients["gpt"])
        elif model == "gemini":
            raw = await call_gemini(persona, frame, clients["gemini"], executor)
        elif model == "claude":
            raw = await call_claude(persona, frame, clients["claude"])
        else:
            raw = None

        if raw is None:
            tqdm.write(f"  [{model:6}] {pair_key} — no result", file=sys.stderr)
            return pair_key, None

        result = {
            **raw,
            "persona_id":      pid,
            "frame_id":        fid,
            "frame_label":     frame["label"],
            "original_frame":  persona["original_frame"],
            "persona_text":    persona["persona"],
            "source_model":    model,
        }

        atomic_write(cache_path, result)

        async with jsonl_lock:
            append_jsonl(jsonl_path, result)

        return pair_key, result


# ---------------------------------------------------------------------------
# Per-model runner across all (persona, frame) pairs
# ---------------------------------------------------------------------------

async def run_model_all_pairs(
    pairs: list[tuple[dict, dict]],
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
            run_model_pair(
                persona, frame, model, clients,
                output_dir, jsonl_path, jsonl_lock, semaphore, executor,
            )
        )
        for persona, frame in pairs
    ]

    with tqdm(
        total=len(tasks),
        desc=f"  [{model:6}]",
        unit="pair",
        dynamic_ncols=True,
        file=sys.stderr,
        colour="cyan",
    ) as pbar:
        for fut in asyncio.as_completed(tasks):
            try:
                pair_key, result = await fut
            except Exception as e:
                tqdm.write(f"  [{model}] unexpected error: {e}", file=sys.stderr)
                pbar.update(1)
                continue
            if result is not None:
                pbar.set_postfix_str(f"last={pair_key}")
                results[pair_key] = result
            else:
                pbar.set_postfix_str(f"last={pair_key} FAILED")
            pbar.update(1)

    return results


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------

async def main_async(args) -> None:
    all_personas = load_personas(PERSONA_CSV)
    all_frames   = load_frames(FRAMES_JSON)

    # Filter by persona ID if requested
    if args.personas:
        wanted = set(args.personas)
        all_personas = [p for p in all_personas if p["persona_id"] in wanted]
        if not all_personas:
            sys.exit(f"[ERROR] no matching persona IDs: {args.personas}")

    # Filter by frame ID (e.g. F01, F07) if requested
    if args.frames:
        wanted_fids = set(f.upper() for f in args.frames)
        all_frames = [f for f in all_frames if f["frame_id"].upper() in wanted_fids]
        if not all_frames:
            sys.exit(f"[ERROR] no matching frame IDs: {args.frames}")

    # Build the cross-product
    pairs: list[tuple[dict, dict]] = [
        (persona, frame)
        for persona in all_personas
        for frame in all_frames
    ]

    models: list[str] = args.models
    output_dir = Path(args.output_dir)

    print(f"Personas    : {len(all_personas)}")
    print(f"Frames      : {len(all_frames)}  "
          f"({', '.join(f['frame_id'] for f in all_frames)})")
    print(f"Total pairs : {len(pairs)} per model")
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
            print(f"  Model : {model.upper()}")
            print(f"  Pairs : {len(pairs)}")
            print(f"{'='*60}")
            results = await run_model_all_pairs(
                pairs, model, clients, output_dir, jsonl_path,
                args.concurrency, executor,
            )
            all_results[model] = results
            n_ok   = len(results)
            n_fail = len(pairs) - n_ok
            print(f"  [{model}] done — {n_ok} ok, {n_fail} failed")
    finally:
        if executor:
            executor.shutdown(wait=False)

    # -- Merge all models into one flat JSON list ----------------------------
    merged: list[dict] = []
    for model in models:
        for rec in all_results.get(model, {}).values():
            merged.append(rec)

    # Sort: model → frame_id → persona_id
    merged.sort(key=lambda r: (r.get("source_model", ""), r.get("frame_id", ""), r.get("persona_id", "")))

    merged_path = output_dir / "merged_frame_persona_cross.json"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nMerged {len(merged)} records → {merged_path}")

    # -- Per-frame × per-model summary --------------------------------------
    from collections import Counter
    print("\n--- Per-frame counts (all models combined) ---")
    frame_counts = Counter(r.get("frame_id", "?") for r in merged)
    for fid in sorted(frame_counts):
        print(f"  {fid}: {frame_counts[fid]}")

    print("\n--- Per-model counts ---")
    model_counts = Counter(r.get("source_model", "?") for r in merged)
    for m, cnt in model_counts.most_common():
        print(f"  {m}: {cnt}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate frame × persona cross-product misogyny comments"
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=["gpt", "gemini", "claude"],
        default=DEFAULT_MODEL_ORDER,
        help="Models to run (default: gpt gemini claude)",
    )
    parser.add_argument(
        "--personas", nargs="+",
        metavar="PERSONA_ID",
        help="Filter to specific persona IDs, e.g. P001 P005 P010",
    )
    parser.add_argument(
        "--frames", nargs="+",
        metavar="FRAME_ID",
        help="Filter to specific JSON frame IDs, e.g. F01 F03 F07",
    )
    parser.add_argument(
        "--concurrency", type=int, default=6,
        help="Max concurrent API calls per model (default: 6)",
    )
    parser.add_argument(
        "--output-dir", default="outputs/frame_persona_cross",
        help="Output directory (default: outputs/frame_persona_cross)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
