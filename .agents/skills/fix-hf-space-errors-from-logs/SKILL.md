---
name: fix-hf-space-errors-from-logs
description: Fetch Hugging Face Space logs for encryptd/ocr_vlm_aggr, identify concrete build/runtime errors, implement the smallest safe fix in the local repo, and validate it. Use this whenever the user asks to debug, triage, or fix Space failures from logs.
---

# Fix HF Space Errors From Logs

## When to use
Use this skill when the user wants to debug `encryptd/ocr_vlm_aggr` by reading Hugging Face Space logs and patching the code based on real errors.

## Inputs
- Space repo id: `encryptd/ocr_vlm_aggr`
- Local repo root: current git repository

## Workflow
1. Verify authentication is available for Hugging Face CLI/API.
   - Prefer existing login (`hf auth whoami`) or `HF_TOKEN` in environment.
   - Never print or echo secret tokens.
2. Collect current Space state and logs.
   - Preferred (new CLI): run `hf spaces logs encryptd/ocr_vlm_aggr --build -n 300` for build errors and `hf spaces logs encryptd/ocr_vlm_aggr -n 300` for runtime/container errors.
   - Fallback (older environments): use Python `huggingface_hub` `HfApi.fetch_space_logs(...)` to fetch build/runtime logs.
   - If one method fails, try the other before stopping; if both fail, report the exact auth/tooling blocker.
3. Persist evidence locally before editing.
   - Save outputs under `logs/hf_space/` with timestamps:
     - `logs/hf_space/<timestamp>_build.log`
     - `logs/hf_space/<timestamp>_runtime.log`
4. Extract actionable failures.
   - Prioritize concrete signatures: `Traceback`, `ModuleNotFoundError`, `ImportError`, `ValueError`, `RuntimeError`, `CUDA out of memory`, `permission denied`, startup/port bind failures.
   - Quote the exact failing message and map it to likely source files.
5. Implement the smallest safe fix in repo code/config.
   - Change only what is necessary to resolve the observed failure.
   - Avoid speculative refactors unrelated to the logged error.
6. Validate locally.
   - Run the most relevant checks available (targeted script/tests first, then broader checks when practical).
   - Confirm the previous error path is addressed.
7. Summarize outcome.
   - State what error was found, what changed conceptually, and what validation passed/failed.
   - If logs are inaccessible due to auth/permissions, stop with a concrete unblock step.

## Important notes
- Treat logs as source-of-truth; do not guess root cause without evidence.
- If multiple errors exist, fix the first blocking one, then re-check logs.
- Keep commits focused and minimal.

## Example prompts
- \"Pull logs for encryptd/ocr_vlm_aggr and fix the crash.\"
- \"Find the Space build error and patch this repo accordingly.\"
