"""
Multi-model synthetic example generator for the casual misogyny annotation study.

Execution order : GPT → Gemini → Claude  (sequential across models, configurable via --models)
Parallelism     : within each model, all frames run concurrently up to --concurrency limit.
                  GPT & Claude use asyncio; Gemini (sync SDK) uses ThreadPoolExecutor.
Resumability    : per-frame atomic JSON cache (rename-swap); stale .tmp files cleaned on
                  startup; Ctrl-C safe — re-run resumes from the last completed frame.
Streaming       : per-model JSONL  outputs/{model}.jsonl — one example per line, flushed
                  and fsynced immediately after each API response.
Progress        : per-model tqdm progress bar — tmux-compatible (stderr, dynamic_ncols).

Usage:
    python generate_examples.py [--models gpt gemini claude] [--frames F01 F02]
                                [--runs 3] [--concurrency 4] [--output-dir outputs]
                                [--gpt-output PATH] [--gemini-output PATH]
                                [--claude-output PATH]

Requires:
    ANTHROPIC_API_KEY  OPENAI_API_KEY  GOOGLE_API_KEY  (in .env or environment)
"""

import argparse
import asyncio
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
GPT_MODEL    = os.getenv("GPT_MODEL",    "gpt-5.4")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TEMPERATURE  = 1.0
MAX_TOKENS   = 8192

DEFAULT_MODEL_ORDER = ["gpt", "gemini", "claude"]

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


def build_user_prompt(frame: dict) -> str:
    fid           = frame["frame_id"]
    label         = frame["label"]
    mtype         = ", ".join(frame.get("misogyny_type", []))
    cov           = frame.get("covertness", "medium")
    belief        = frame.get("core_belief", "")
    manifestations = "\n".join(f"  - {m}" for m in frame.get("surface_manifestations", []))
    ling_cues     = "\n".join(f"  - {c}" for c in frame.get("linguistic_cues", []))
    india_ctx     = frame.get("india_specific_context", "")
    hints         = "\n".join(f"  - {h}" for h in frame.get("llm_generation_prompt_hints", []))

    return f"""
Frame to generate examples for:

Frame ID: {fid}
Frame Label: {label}
Misogyny Type: {mtype}
Covertness Level: {cov}

Core Belief (encode this worldview — do NOT state it explicitly):
{belief}

Surface Manifestations:
{manifestations}

Linguistic Cues to use:
{ling_cues}

India-specific Context:
{india_ctx}

Generation hints:
{hints}

---

TASK: Generate exactly 12 examples for frame {fid} across the 5 scenario types:
  1. whatsapp          — informal, code-switched Hindi-English, short
  2. reddit_quora      — semi-formal, argumentative
  3. spoken_convo      — 2–3 turn dialogue, casual
  4. social_media      — Twitter/Instagram style
  5. family_workplace  — contextual, 1–2 sentences

For scenario types 1–4: 1 COVERT + 1 OVERT example each.
For scenario type 5   : 1 COVERT + 1 OVERT example.
Plus exactly 2 NEAR-MISS examples (NOT misogynistic, superficially similar), label = "non_misogynistic".

Speaker gender: ~30 % of examples should have speaker_gender = "woman".

Rules:
- Speaker must NOT sound like they know they are being misogynistic
- No slurs, no graphic content — survey-safe
- Avoid strawmanning — the misogynistic position must sound genuinely held
- Use Indian names, places, and register
- For spoken_convo use "Speaker A: / Speaker B:" labels inside the text field

Output ONLY a valid JSON object (no markdown fences, no prose):
{{
  "frame_id": "{fid}",
  "frame_label": "{label}",
  "misogyny_type": {json.dumps(frame.get("misogyny_type", []))},
  "covertness_level": "{cov}",
  "examples": [
    {{
      "id": "{fid}_001",
      "scenario_type": "whatsapp",
      "covertness": "covert",
      "speaker_gender": "man",
      "text": "...",
      "plausible_deniability": "concern_defense",
      "label": "misogynistic",
      "annotation_note": "brief note on why this is misogynistic"
    }}
  ]
}}

Number ids sequentially: {fid}_001 … {fid}_012.
""".strip()


# ---------------------------------------------------------------------------
# Helpers
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


def deduplicate_examples(examples: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for ex in examples:
        key = ex.get("text", "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def atomic_write(path: Path, data: dict) -> None:
    """Write JSON to a .tmp file then atomically rename — safe against mid-write crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def validate_cache(path: Path) -> dict | None:
    """Load cached frame JSON; return None (and delete) if missing or corrupt."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "examples" not in data or not isinstance(data["examples"], list):
            raise ValueError("missing 'examples' list")
        if not data["examples"]:
            raise ValueError("empty examples list")
        return data
    except Exception as e:
        tqdm.write(f"  [cache] corrupt {path.name}: {e} — will regenerate", file=sys.stderr)
        path.unlink(missing_ok=True)
        path.with_suffix(".tmp").unlink(missing_ok=True)
        return None


def append_jsonl(path: Path, records: list[dict]) -> None:
    """Append records to a JSONL file and fsync — called while holding jsonl_lock."""
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
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

async def call_claude(frame: dict, client: anthropic.AsyncAnthropic) -> dict | None:
    prompt = build_user_prompt(frame)
    for system in (SYSTEM_PROMPT, SYSTEM_PROMPT_FALLBACK):
        try:
            async with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                thinking={"type": "adaptive"},
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
            return await call_claude(frame, client)
        except Exception as e:
            tqdm.write(f"  [claude] error: {e}", file=sys.stderr)
            return None
    return None


def _gpt_kwargs(use_completion_tokens: bool) -> dict:
    """Build GPT API kwargs, switching to max_completion_tokens for newer models."""
    kwargs: dict = {"model": GPT_MODEL}
    if use_completion_tokens:
        kwargs["max_completion_tokens"] = MAX_TOKENS
        # o-series models do not support temperature
    else:
        kwargs["max_tokens"] = MAX_TOKENS
        kwargs["temperature"] = TEMPERATURE
    return kwargs


async def call_gpt(frame: dict, client: AsyncOpenAI) -> dict | None:
    prompt = build_user_prompt(frame)
    use_completion_tokens = False  # try legacy param first; flip on 400
    for system in (SYSTEM_PROMPT, SYSTEM_PROMPT_FALLBACK):
        try:
            resp = await client.chat.completions.create(
                **_gpt_kwargs(use_completion_tokens),
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
                return await call_gpt(frame, client)
            if "max_tokens" in msg and "max_completion_tokens" in msg:
                # Model requires max_completion_tokens — retry immediately with it
                tqdm.write("  [gpt] switching to max_completion_tokens", file=sys.stderr)
                use_completion_tokens = True
                continue
            if system == SYSTEM_PROMPT:
                tqdm.write(f"  [gpt] retrying with fallback prompt: {e}", file=sys.stderr)
                continue
            tqdm.write(f"  [gpt] error: {e}", file=sys.stderr)
            return None
    return None


def _gemini_sync(frame: dict, model) -> dict | None:
    """Synchronous Gemini call — dispatched via ThreadPoolExecutor."""
    for attempt, sys_p in enumerate((SYSTEM_PROMPT, SYSTEM_PROMPT_FALLBACK)):
        prompt = sys_p + "\n\n" + build_user_prompt(frame)
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
                return _gemini_sync(frame, model)
            tqdm.write(f"  [gemini] error (attempt {attempt + 1}): {e}", file=sys.stderr)
            if attempt == 0:
                time.sleep(3)
    return None


async def call_gemini(frame: dict, model, executor: ThreadPoolExecutor) -> dict | None:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _gemini_sync, frame, model)


CALLERS = {
    "gpt":    call_gpt,
    "gemini": call_gemini,
    "claude": call_claude,
}


# ---------------------------------------------------------------------------
# Per-frame, per-model runner  (bounded by a shared semaphore)
# ---------------------------------------------------------------------------

async def run_model_frame(
    frame: dict,
    model: str,
    runs: int,
    clients: dict,
    output_dir: Path,
    jsonl_path: Path,
    jsonl_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor | None,
) -> tuple[str, dict | None]:
    """Process one frame for one model; return (frame_id, result_or_None)."""
    fid        = frame["frame_id"]
    cache_path = output_dir / model / f"{fid}.json"

    # Fast path: already done
    async with semaphore:
        cached = validate_cache(cache_path)
        if cached:
            n = len(cached["examples"])
            tqdm.write(f"  [{model:6}] {fid} — cache hit ({n} examples)", file=sys.stderr)
            return fid, cached

        all_examples: list[dict] = []
        base_meta:    dict        = {}

        for run in range(1, runs + 1):
            if model == "gpt":
                raw = await call_gpt(frame, clients["gpt"])
            elif model == "gemini":
                raw = await call_gemini(frame, clients["gemini"], executor)
            elif model == "claude":
                raw = await call_claude(frame, clients["claude"])
            else:
                raw = None

            if raw is None:
                tqdm.write(f"  [{model:6}] {fid} run {run}/{runs} — no result", file=sys.stderr)
                continue

            if not base_meta:
                base_meta = {k: v for k, v in raw.items() if k != "examples"}

            new_exs = raw.get("examples", [])
            all_examples.extend(new_exs)

            # Stream examples to per-model JSONL immediately with fsync
            async with jsonl_lock:
                append_jsonl(
                    jsonl_path,
                    [{**ex, "frame_id": fid, "source_model": model} for ex in new_exs],
                )

            if run < runs:
                await asyncio.sleep(2)  # small inter-run backoff

        if not base_meta:
            return fid, None

        unique = deduplicate_examples(all_examples)
        for i, ex in enumerate(unique, 1):
            ex["id"] = f"{fid}_{i:03d}"

        result = {**base_meta, "examples": unique}
        atomic_write(cache_path, result)
        return fid, result


# ---------------------------------------------------------------------------
# Per-model runner across all frames  (parallel with tqdm progress bar)
# ---------------------------------------------------------------------------

async def run_model_all_frames(
    frames: list[dict],
    model: str,
    runs: int,
    clients: dict,
    output_dir: Path,
    jsonl_path: Path,
    concurrency: int,
    executor: ThreadPoolExecutor | None,
) -> dict[str, dict]:
    """Run all frames for one model in parallel (bounded by concurrency semaphore)."""
    semaphore  = asyncio.Semaphore(concurrency)
    jsonl_lock = asyncio.Lock()
    results: dict[str, dict] = {}

    tasks = [
        asyncio.create_task(
            run_model_frame(
                frame, model, runs, clients,
                output_dir, jsonl_path, jsonl_lock, semaphore, executor,
            )
        )
        for frame in frames
    ]

    with tqdm(
        total=len(tasks),
        desc=f"  [{model:6}]",
        unit="frame",
        dynamic_ncols=True,
        file=sys.stderr,
        colour="cyan",
    ) as pbar:
        for fut in asyncio.as_completed(tasks):
            try:
                fid, result = await fut
            except Exception as e:
                tqdm.write(f"  [{model}] unexpected error: {e}", file=sys.stderr)
                pbar.update(1)
                continue
            if result is not None:
                n = len(result.get("examples", []))
                pbar.set_postfix_str(f"last={fid} {n}ex")
                results[fid] = result
            else:
                pbar.set_postfix_str(f"last={fid} FAILED")
            pbar.update(1)

    return results


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------

async def main_async(args) -> None:
    # -- Load frames ---------------------------------------------------------
    frames_path = Path(args.frames_file)
    if not frames_path.exists():
        sys.exit(f"[ERROR] frames file not found: {frames_path}")
    data = json.loads(frames_path.read_text(encoding="utf-8"))
    all_frames: list[dict] = data["frames"]

    if args.frames:
        all_frames = [f for f in all_frames if f["frame_id"] in args.frames]
        if not all_frames:
            sys.exit(f"[ERROR] no matching frames: {args.frames}")

    models: list[str] = args.models
    output_dir = Path(args.output_dir)

    print(f"Frames      : {[f['frame_id'] for f in all_frames]}")
    print(f"Models      : {' → '.join(models)}  (sequential)")
    print(f"Runs/model  : {args.runs}   Concurrency per model: {args.concurrency}")

    # -- Startup: clean any stale .tmp files from a crashed previous run -----
    cleanup_tmp_files(output_dir, models)

    # -- Per-model JSONL output paths (default: outputs/{model}.jsonl) -------
    model_jsonl: dict[str, Path] = {
        "gpt":    Path(args.gpt_output)    if args.gpt_output    else output_dir / "gpt.jsonl",
        "gemini": Path(args.gemini_output) if args.gemini_output else output_dir / "gemini.jsonl",
        "claude": Path(args.claude_output) if args.claude_output else output_dir / "claude.jsonl",
    }

    # -- Output dirs ---------------------------------------------------------
    for model in models:
        (output_dir / model).mkdir(parents=True, exist_ok=True)

    # -- API clients ---------------------------------------------------------
    clients: dict = {}
    if "gpt" in models:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("[ERROR] OPENAI_API_KEY not set")
        clients["gpt"] = AsyncOpenAI(api_key=key)

    if "gemini" in models:
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            sys.exit("[ERROR] GOOGLE_API_KEY not set")
        genai.configure(api_key=key)
        clients["gemini"] = genai.GenerativeModel(GEMINI_MODEL)

    if "claude" in models:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            sys.exit("[ERROR] ANTHROPIC_API_KEY not set")
        clients["claude"] = anthropic.AsyncAnthropic(api_key=key)

    # Dedicated thread-pool for Gemini's sync SDK
    gemini_executor = (
        ThreadPoolExecutor(max_workers=args.concurrency) if "gemini" in models else None
    )

    # -- Process models sequentially: gpt → gemini → claude ------------------
    per_model: dict[str, dict[str, dict]] = {}
    try:
        for model in models:
            print(f"\n{'='*60}")
            print(f"  Model : {model.upper()}  —  {len(all_frames)} frames  "
                  f"(≤{args.concurrency} parallel)")
            print(f"  Output: {model_jsonl[model]}")
            print(f"{'='*60}")
            per_model[model] = await run_model_all_frames(
                all_frames, model, args.runs, clients,
                output_dir, model_jsonl[model],
                args.concurrency, gemini_executor,
            )
    except KeyboardInterrupt:
        tqdm.write("\n[interrupted] progress saved — re-run to resume", file=sys.stderr)
        return
    finally:
        if gemini_executor:
            gemini_executor.shutdown(wait=False)

    # -- Per-model aggregated JSON output ------------------------------------
    for model in models:
        frames_list = list(per_model[model].values())
        out = output_dir / f"{model}_all_frames.json"
        atomic_write(out, frames_list)
        n_ex = sum(len(f.get("examples", [])) for f in frames_list)
        print(f"\n[{model}] {len(frames_list)} frames, {n_ex} examples → {out}")
        print(f"[{model}] stream → {model_jsonl[model]}")

    # -- Merged output -------------------------------------------------------
    merged: list[dict] = []
    for frame in all_frames:
        fid = frame["frame_id"]
        all_exs: list[dict] = []
        base: dict = {}
        for model in models:
            fd = per_model[model].get(fid)
            if not fd:
                continue
            if not base:
                base = {k: v for k, v in fd.items() if k != "examples"}
            for ex in fd.get("examples", []):
                all_exs.append({**ex, "source_model": model})
        if base:
            merged.append({**base, "examples": all_exs})

    merged_out = output_dir / "merged_all_models.json"
    atomic_write(merged_out, merged)
    n_merged = sum(len(f.get("examples", [])) for f in merged)
    print(f"\n[merged] {len(merged)} frames, {n_merged} total examples → {merged_out}")

    # -- Summary -------------------------------------------------------------
    print("\n=== Summary ===")
    for model in models:
        frames_list = list(per_model[model].values())
        n_ex  = sum(len(f.get("examples", [])) for f in frames_list)
        n_mis = sum(1 for f in frames_list for ex in f.get("examples", [])
                    if ex.get("label") == "misogynistic")
        n_non = sum(1 for f in frames_list for ex in f.get("examples", [])
                    if ex.get("label") == "non_misogynistic")
        print(f"  {model:10}: {n_ex:4d} examples  "
              f"({n_mis} misogynistic, {n_non} non-misogynistic)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate misogyny annotation examples via LLMs"
    )
    parser.add_argument("--frames-file",    default="misogyny_frames.json")
    parser.add_argument("--models",         nargs="+", default=DEFAULT_MODEL_ORDER,
                        choices=["gpt", "gemini", "claude"])
    parser.add_argument("--frames",         nargs="*",
                        help="Specific frame IDs to generate (default: all)")
    parser.add_argument("--runs",           type=int, default=3)
    parser.add_argument("--concurrency",    type=int, default=4,
                        help="Max parallel frames per model (default: 4)")
    parser.add_argument("--output-dir",     default="outputs")
    parser.add_argument("--gpt-output",     default=None,
                        metavar="PATH",
                        help="JSONL stream for GPT   (default: outputs/gpt.jsonl)")
    parser.add_argument("--gemini-output",  default=None,
                        metavar="PATH",
                        help="JSONL stream for Gemini (default: outputs/gemini.jsonl)")
    parser.add_argument("--claude-output",  default=None,
                        metavar="PATH",
                        help="JSONL stream for Claude (default: outputs/claude.jsonl)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
