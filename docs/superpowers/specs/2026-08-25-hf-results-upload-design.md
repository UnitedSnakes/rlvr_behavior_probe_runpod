# Hugging Face Results Upload Design

Date: 2026-08-25

## Goal

Add an opt-in results backup path to `run_probe.py` so a completed experiment can automatically upload its local result directory to a pre-existing Hugging Face Dataset repository.

The default experiment workflow must remain unchanged when upload is not requested.

## Scope

This change adds one optional CLI argument to `run_probe.py`:

```text
--upload-repo <owner/repo>
```

Example:

```bash
python run_probe.py \
  --engine vllm \
  --result-dir results_sft256_vllm \
  --upload-repo UnitedSnakes/rlvr-behavior-probe-results
```

When the flag is omitted, no Hugging Face upload code is invoked.

The implementation will add a focused upload helper under `probe/` and tests for upload-related behavior. It will not create Hugging Face repositories, manage repository visibility, upload model checkpoints, or change rollout/scoring semantics.

## Architecture

`run_probe.py` remains responsible for experiment orchestration. Hugging Face API details live in a separate helper module, tentatively:

```text
probe/results_upload.py
```

The flow is:

```text
run starts
→ record UTC run start time
→ execute SFT/RL sampling as usual
→ write local result files
→ write run_config.json
→ if --upload-repo is set:
     validate upload configuration
     upload the entire result directory
→ exit successfully only if requested upload succeeds
```

Separating the uploader from `run_probe.py` keeps the experiment script small and makes the upload logic independently testable and reusable for a future manual retry path.

## Run identity and timestamp

Capture the run start time once, at the beginning of `main()`, before sampling begins.

Use an offset-aware UTC timestamp. Keep two representations:

1. Human/machine-readable ISO form for config, for example:

```text
2026-08-25T23:53:12Z
```

2. Filesystem/Hub-safe compact form for the remote path:

```text
20260825T235312Z
```

The timestamp must not be recomputed at upload time. A long experiment therefore keeps the same run identity from start to finish.

## Remote layout

Upload each result directory beneath:

```text
runs/<UTC-start-timestamp>-<result-dir-name>/
```

Example:

```text
runs/20260825T235312Z-results_sft256_vllm/
```

The local `result_dir` basename is used as the experiment label. Only the basename is included in the remote path; parent directories are not reproduced.

The uploader must refuse to overwrite an already existing remote run directory. An unexpected collision is treated as an error rather than silently replacing prior experiment data.

## Hugging Face repository behavior

The destination repository must already exist.

The program only uploads to the repository supplied through `--upload-repo`; it must not call repository-creation APIs.

The intended repository type is `dataset`.

A typical destination is:

```text
UnitedSnakes/rlvr-behavior-probe-results
```

Repository privacy/publicity remains a Hugging Face-side setting and is outside this program's responsibility.

## Authentication

Authentication comes only from the environment variable:

```text
HF_TOKEN
```

There is no CLI token argument.

The token must never be written to:

- `run_config.json`
- stdout/stderr
- uploaded files
- Git history

If `--upload-repo` is set but `HF_TOKEN` is absent or empty, the command fails before attempting an upload and reports a clear authentication error.

On RunPod, the intended template configuration is to inject `HF_TOKEN` from a RunPod Secret at runtime.

## Local outputs and config metadata

Existing local experiment outputs remain authoritative until upload succeeds.

`run_config.json` should additionally record:

```json
{
  "run_started_at": "2026-08-25T23:53:12Z",
  "upload_repo": "UnitedSnakes/rlvr-behavior-probe-results",
  "upload_path": "runs/20260825T235312Z-results_sft256_vllm"
}
```

When no upload is requested:

- `run_started_at` is still recorded.
- `upload_repo` is null.
- `upload_path` is null.

No token or other credential metadata is recorded.

## Upload contents

When requested, upload the entire `result_dir` after the local run has completed and `run_config.json` has been written.

This includes the raw rollout files and config produced by the run, such as:

```text
sft_raw.jsonl
rl_raw.jsonl
run_config.json
```

For SFT-only or RL-only runs, upload whatever files that mode normally produces. Upload behavior does not require paired SFT/RL output.

The uploader should use `huggingface_hub.HfApi().upload_folder()` or the current equivalent supported by the pinned environment.

## Failure semantics

Local experiment results must never be deleted because of an upload failure.

If sampling or local result writing fails, no upload is attempted.

If sampling and local writing succeed but the requested upload fails, the process returns non-zero. The error should make it clear that local results exist but remote backup did not complete.

Examples of upload failures include:

- missing `HF_TOKEN`
- destination repository does not exist or is inaccessible
- remote run-path collision
- authentication failure
- network failure
- Hugging Face API error

The implementation must not catch these failures and then report overall success.

## Existing behavior preservation

Without `--upload-repo`, `run_probe.py` must preserve current behavior:

- same sampling flow
- same model/revision resolution
- same raw output format
- same resume behavior
- same SFT-only/RL-only behavior
- no Hugging Face network call for result upload

The only unconditional metadata addition is `run_started_at` in `run_config.json`.

## Testing

Unit tests should mock the Hugging Face API. Tests must not perform real uploads or depend on network access.

At minimum, cover:

1. Remote path generation uses the run-start UTC timestamp and local result directory basename.
2. The timestamp is stable for the entire run and is not regenerated at upload time.
3. Omitting `--upload-repo` never invokes the upload helper.
4. `run_config.json` records `run_started_at`, `upload_repo`, and `upload_path` correctly.
5. Requested upload without `HF_TOKEN` fails clearly.
6. Requested upload calls the Hugging Face API with repository type `dataset`, the expected repo ID, local folder, and remote path.
7. Upload/API exceptions propagate so the command exits unsuccessfully.
8. Local result files remain present when upload fails.
9. An existing remote destination path is rejected rather than overwritten.
10. SFT-only and RL-only runs can upload their result directories without requiring paired output files.

Existing runtime/mode tests should continue to pass.

## Documentation

Update the README with one short usage example showing:

- `HF_TOKEN` supplied through the environment
- `--upload-repo`
- resulting `runs/<timestamp>-<result-dir-name>/` layout

The README must not include a real token.

## Non-goals

This design does not attempt to:

- create or configure Hugging Face repositories
- make upload mandatory
- upload partial/in-progress runs
- continuously sync during generation
- retry indefinitely after network failure
- delete local files after successful upload
- store Hugging Face credentials in CLI arguments or config files
- add RunPod bootstrap/init logic
- preserve `/workspace` through a Network Volume

A standalone manual re-upload command may be added later, reusing the same helper, but is not required for this change.

## Success criteria

A normal run without `--upload-repo` behaves as before and performs no result-upload network call.

A run with:

```text
--upload-repo UnitedSnakes/rlvr-behavior-probe-results
```

and a valid `HF_TOKEN` writes all local outputs first, then uploads the complete result directory to a unique path such as:

```text
runs/20260825T235312Z-results_sft256_vllm/
```

If remote backup fails, the local results remain available and the overall command reports failure rather than falsely signaling that the run is fully backed up.
