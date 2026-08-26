# Hugging Face Results Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `--upload-repo` path to `run_probe.py` that uploads a completed local result directory to a unique timestamped folder in a pre-existing Hugging Face Dataset repository and fails loudly if the requested backup does not complete.

**Architecture:** Keep Hugging Face API logic in a focused `probe/results_upload.py` helper. `run_probe.py` records one UTC start time, derives the remote run path from that fixed timestamp and the local result-directory basename, writes all local outputs and config first, then invokes the helper only when `--upload-repo` is set. Upload failures preserve local files and propagate as a non-zero process outcome.

**Tech Stack:** Python 3.12, `huggingface_hub>=0.24`, `argparse`, `pathlib`, `datetime`, `pytest`, `monkeypatch`.

**Spec:** `docs/superpowers/specs/2026-08-25-hf-results-upload-design.md`

## Global Constraints

- Upload is opt-in through `--upload-repo <owner/repo>`; omitting the flag must make no result-upload network call.
- Destination repositories must already exist; implementation must never create a Hugging Face repository.
- Authentication comes only from environment variable `HF_TOKEN`; there is no CLI token argument and the token is never written to config or logs.
- Run identity is the UTC experiment start time, captured once before sampling and reused for config and remote-path generation.
- Remote layout is `runs/<UTC-start-timestamp>-<result-dir-name>/` using only the local directory basename.
- Existing remote run paths must be rejected rather than overwritten.
- Upload occurs only after sampling/local writing and `run_config.json` complete successfully.
- Upload failure must leave local files untouched and make the overall command fail.
- `run_config.json` always records `run_started_at`; `upload_repo` and `upload_path` are null when upload is not requested.
- SFT-only and RL-only modes remain valid upload sources; paired output files are not required.
- Do not add RunPod bootstrap/init behavior in this change.

---

### Task 1: Add an isolated Hugging Face results uploader

**Files:**
- Create: `probe/results_upload.py`
- Create: `tests/test_results_upload.py`

**Interfaces:**
- Produces: `build_remote_run_path(result_dir: Path, run_started_at: datetime) -> str`
- Produces: `format_run_started_at(run_started_at: datetime) -> str`
- Produces: `upload_result_dir(result_dir: Path, repo_id: str, remote_path: str) -> None`
- `upload_result_dir` reads `HF_TOKEN` internally, creates `HfApi(token=token)`, checks the Dataset repository for a conflicting remote prefix, then calls `upload_folder(...)`.

- [ ] **Step 1: Write failing tests for timestamp formatting and remote-path generation**

Create `tests/test_results_upload.py` with deterministic UTC datetimes:

```python
from datetime import datetime, timezone
from pathlib import Path

from probe.results_upload import build_remote_run_path, format_run_started_at


def test_format_run_started_at_uses_utc_z_suffix():
    started_at = datetime(2026, 8, 25, 23, 53, 12, tzinfo=timezone.utc)

    assert format_run_started_at(started_at) == "2026-08-25T23:53:12Z"


def test_build_remote_run_path_uses_start_time_and_result_dir_basename():
    started_at = datetime(2026, 8, 25, 23, 53, 12, tzinfo=timezone.utc)
    result_dir = Path("/workspace/experiments/results_sft256_vllm")

    assert build_remote_run_path(result_dir, started_at) == (
        "runs/20260825T235312Z-results_sft256_vllm"
    )
```

- [ ] **Step 2: Run the new tests and verify they fail because the module does not exist yet**

Run:

```bash
python -m pytest -q tests/test_results_upload.py
```

Expected: collection/import failure for `probe.results_upload`.

- [ ] **Step 3: Implement the timestamp/path helpers minimally**

Create `probe/results_upload.py` with timezone validation and stable formatting:

```python
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


def _as_utc(run_started_at: datetime) -> datetime:
    if run_started_at.tzinfo is None:
        raise ValueError("run_started_at must be timezone-aware")
    return run_started_at.astimezone(timezone.utc)


def format_run_started_at(run_started_at: datetime) -> str:
    utc_time = _as_utc(run_started_at).replace(microsecond=0)
    return utc_time.isoformat().replace("+00:00", "Z")


def build_remote_run_path(result_dir: Path, run_started_at: datetime) -> str:
    utc_time = _as_utc(run_started_at).replace(microsecond=0)
    timestamp = utc_time.strftime("%Y%m%dT%H%M%SZ")
    return f"runs/{timestamp}-{result_dir.name}"
```

Do not implement upload yet.

- [ ] **Step 4: Run the timestamp/path tests and verify they pass**

Run:

```bash
python -m pytest -q tests/test_results_upload.py
```

Expected: 2 tests pass.

- [ ] **Step 5: Add failing tests for authentication, Dataset API arguments, collision rejection, and API error propagation**

Extend `tests/test_results_upload.py`. Use a fake `HfApi` so no network request occurs:

```python
import pytest

import probe.results_upload as results_upload


class FakeApi:
    def __init__(self, token):
        self.token = token
        self.upload_calls = []
        self.repo_files = []
        self.upload_error = None

    def list_repo_files(self, *, repo_id, repo_type):
        assert repo_type == "dataset"
        return list(self.repo_files)

    def upload_folder(self, **kwargs):
        if self.upload_error is not None:
            raise self.upload_error
        self.upload_calls.append(kwargs)


def test_upload_requires_hf_token(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        results_upload.upload_result_dir(
            tmp_path,
            "UnitedSnakes/rlvr-behavior-probe-results",
            "runs/20260825T235312Z-results",
        )


def test_upload_uses_dataset_repo_and_expected_remote_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    fake_api = FakeApi(token="test-token")
    monkeypatch.setattr(results_upload, "HfApi", lambda token: fake_api)

    results_upload.upload_result_dir(
        tmp_path,
        "UnitedSnakes/rlvr-behavior-probe-results",
        "runs/20260825T235312Z-results",
    )

    assert fake_api.upload_calls == [
        {
            "folder_path": str(tmp_path),
            "repo_id": "UnitedSnakes/rlvr-behavior-probe-results",
            "repo_type": "dataset",
            "path_in_repo": "runs/20260825T235312Z-results",
        }
    ]


def test_upload_rejects_existing_remote_run(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    fake_api = FakeApi(token="test-token")
    fake_api.repo_files = [
        "runs/20260825T235312Z-results/run_config.json",
    ]
    monkeypatch.setattr(results_upload, "HfApi", lambda token: fake_api)

    with pytest.raises(FileExistsError, match="already exists"):
        results_upload.upload_result_dir(
            tmp_path,
            "UnitedSnakes/rlvr-behavior-probe-results",
            "runs/20260825T235312Z-results",
        )

    assert fake_api.upload_calls == []


def test_upload_api_errors_propagate(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    fake_api = FakeApi(token="test-token")
    fake_api.upload_error = RuntimeError("network down")
    monkeypatch.setattr(results_upload, "HfApi", lambda token: fake_api)

    with pytest.raises(RuntimeError, match="network down"):
        results_upload.upload_result_dir(
            tmp_path,
            "UnitedSnakes/rlvr-behavior-probe-results",
            "runs/20260825T235312Z-results",
        )
```

Also assert in the successful test that the fake API received the environment token; do not print or persist the token.

- [ ] **Step 6: Run the uploader tests and verify the new cases fail because `upload_result_dir` is missing**

Run:

```bash
python -m pytest -q tests/test_results_upload.py
```

Expected: timestamp/path tests pass; upload tests fail with missing `upload_result_dir`.

- [ ] **Step 7: Implement `upload_result_dir` with collision detection and no repo creation**

Add to `probe/results_upload.py`:

```python
def upload_result_dir(result_dir: Path, repo_id: str, remote_path: str) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required when --upload-repo is requested"
        )

    api = HfApi(token=token)
    existing_files = api.list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
    )

    normalized_remote_path = remote_path.rstrip("/")
    remote_prefix = normalized_remote_path + "/"
    collision = any(
        file_path == normalized_remote_path
        or file_path.startswith(remote_prefix)
        for file_path in existing_files
    )
    if collision:
        raise FileExistsError(
            f"Remote run path already exists: {normalized_remote_path}"
        )

    api.upload_folder(
        folder_path=str(result_dir),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=normalized_remote_path,
    )
```

Do not call `create_repo()` or add retry/overwrite behavior.

- [ ] **Step 8: Run uploader tests and verify all pass**

Run:

```bash
python -m pytest -q tests/test_results_upload.py
```

Expected: all uploader tests pass.

- [ ] **Step 9: Commit the isolated uploader**

```bash
git add probe/results_upload.py tests/test_results_upload.py
git commit -m "feat: add Hugging Face results uploader"
```

---

### Task 2: Integrate opt-in upload into `run_probe.py`

**Files:**
- Modify: `run_probe.py`
- Modify: `tests/test_run_probe_modes.py`

**Interfaces:**
- Consumes: `build_remote_run_path(result_dir: Path, run_started_at: datetime) -> str`
- Consumes: `format_run_started_at(run_started_at: datetime) -> str`
- Consumes: `upload_result_dir(result_dir: Path, repo_id: str, remote_path: str) -> None`
- Adds CLI argument: `--upload-repo`, default `None`.
- `run_config.json` gains `run_started_at`, `upload_repo`, and `upload_path`.

- [ ] **Step 1: Add failing parser/config tests for the opt-in flag and no-upload default**

Extend `tests/test_run_probe_modes.py` with tests that monkeypatch runtime-heavy functions as existing tests already do.

First verify the parser default:

```python
def test_upload_repo_defaults_to_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_probe.py"])

    args = run_probe.parse_args()

    assert args.upload_repo is None
```

Then add a no-upload orchestration test. Freeze start time by monkeypatching a small `utc_now()` helper that Task 2 will add to `run_probe.py`:

```python
from datetime import datetime, timezone
import json


def test_no_upload_records_start_time_and_never_calls_uploader(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_probe.py",
            "--only-rl",
            "--result-dir",
            str(tmp_path),
        ],
    )
    started_at = datetime(2026, 8, 25, 23, 53, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(run_probe, "utc_now", lambda: started_at)
    monkeypatch.setattr(run_probe, "set_seed", lambda seed: None)
    monkeypatch.setattr(run_probe, "resolve_device", lambda device: "cuda")
    monkeypatch.setattr(run_probe, "resolve_dtype", lambda dtype: "bfloat16")
    monkeypatch.setattr(
        run_probe,
        "prepare_questions",
        lambda path, questions, seed: [
            {"qid": 0, "question": "1+1?", "gold": "2"}
        ],
    )
    monkeypatch.setattr(run_probe, "run_one_checkpoint", lambda **kwargs: None)

    upload_calls = []
    monkeypatch.setattr(
        run_probe,
        "upload_result_dir",
        lambda **kwargs: upload_calls.append(kwargs),
    )

    run_probe.main()

    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["run_started_at"] == "2026-08-25T23:53:12Z"
    assert config["upload_repo"] is None
    assert config["upload_path"] is None
    assert upload_calls == []
```

- [ ] **Step 2: Run the focused tests and verify they fail on missing parser/helper integration**

Run:

```bash
python -m pytest -q \
  tests/test_run_probe_modes.py::test_upload_repo_defaults_to_none \
  tests/test_run_probe_modes.py::test_no_upload_records_start_time_and_never_calls_uploader
```

Expected: failures because `--upload-repo`, `utc_now`, and uploader imports are not wired yet.

- [ ] **Step 3: Add minimal CLI, clock helper, metadata calculation, and imports**

In `run_probe.py`:

```python
from datetime import datetime, timezone

from probe.results_upload import (
    build_remote_run_path,
    format_run_started_at,
    upload_result_dir,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)
```

Add under the I/O arguments:

```python
parser.add_argument(
    "--upload-repo",
    default=None,
    help=(
        "Optional pre-existing Hugging Face Dataset repo to receive the "
        "completed result directory. Authentication uses HF_TOKEN."
    ),
)
```

At the beginning of `main()` immediately after parsing arguments:

```python
run_started_at = utc_now()
```

After `result_dir` is created, derive upload metadata exactly once:

```python
if args.upload_repo:
    upload_path = build_remote_run_path(result_dir, run_started_at)
else:
    upload_path = None
```

Before writing `run_config.json`:

```python
config["run_started_at"] = format_run_started_at(run_started_at)
config["upload_repo"] = args.upload_repo
config["upload_path"] = upload_path
```

Do not invoke upload yet in this step.

- [ ] **Step 4: Run the parser/no-upload tests and verify they pass**

Run:

```bash
python -m pytest -q \
  tests/test_run_probe_modes.py::test_upload_repo_defaults_to_none \
  tests/test_run_probe_modes.py::test_no_upload_records_start_time_and_never_calls_uploader
```

Expected: both pass.

- [ ] **Step 5: Add failing tests for successful upload, stable start timestamp, upload-after-config ordering, and failure preservation**

Add a successful upload integration test:

```python
def test_requested_upload_runs_after_config_is_written(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_probe.py",
            "--only-rl",
            "--result-dir",
            str(tmp_path),
            "--upload-repo",
            "UnitedSnakes/rlvr-behavior-probe-results",
        ],
    )
    started_at = datetime(2026, 8, 25, 23, 53, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(run_probe, "utc_now", lambda: started_at)
    monkeypatch.setattr(run_probe, "set_seed", lambda seed: None)
    monkeypatch.setattr(run_probe, "resolve_device", lambda device: "cuda")
    monkeypatch.setattr(run_probe, "resolve_dtype", lambda dtype: "bfloat16")
    monkeypatch.setattr(
        run_probe,
        "prepare_questions",
        lambda path, questions, seed: [
            {"qid": 0, "question": "1+1?", "gold": "2"}
        ],
    )

    def fake_run_one_checkpoint(**kwargs):
        kwargs["out_path"].write_text("local result\n", encoding="utf-8")

    monkeypatch.setattr(run_probe, "run_one_checkpoint", fake_run_one_checkpoint)

    upload_calls = []

    def fake_upload_result_dir(result_dir, repo_id, remote_path):
        assert (result_dir / "run_config.json").exists()
        assert (result_dir / "rl_raw.jsonl").exists()
        upload_calls.append((result_dir, repo_id, remote_path))

    monkeypatch.setattr(run_probe, "upload_result_dir", fake_upload_result_dir)

    run_probe.main()

    assert upload_calls == [
        (
            tmp_path,
            "UnitedSnakes/rlvr-behavior-probe-results",
            f"runs/20260825T235312Z-{tmp_path.name}",
        )
    ]
```

Add a failure test that proves local files remain and the exception is not swallowed:

```python
def test_upload_failure_preserves_local_results_and_fails_run(monkeypatch, tmp_path):
    # Reuse the same lightweight runtime monkeypatch pattern as the successful
    # upload test, with --upload-repo enabled and a fixed utc_now().
    def fake_run_one_checkpoint(**kwargs):
        kwargs["out_path"].write_text("local result\n", encoding="utf-8")

    monkeypatch.setattr(run_probe, "run_one_checkpoint", fake_run_one_checkpoint)
    monkeypatch.setattr(
        run_probe,
        "upload_result_dir",
        lambda result_dir, repo_id, remote_path: (_ for _ in ()).throw(
            RuntimeError("network down")
        ),
    )

    with pytest.raises(RuntimeError, match="local results.*backup failed"):
        run_probe.main()

    assert (tmp_path / "rl_raw.jsonl").exists()
    assert (tmp_path / "run_config.json").exists()
```

In the actual test file, include the full argv/runtime monkeypatch setup rather than relying on hidden shared state. Assert that the `upload_path` inside `run_config.json` exactly matches the path passed to the uploader, proving the same start timestamp is reused.

Add one parameterized test over `--only-sft` and `--only-rl` showing either mode can reach the upload call without requiring the other raw file.

- [ ] **Step 6: Run the new integration tests and verify they fail because upload is not invoked yet**

Run:

```bash
python -m pytest -q tests/test_run_probe_modes.py -k "upload"
```

Expected: metadata-only tests pass; requested-upload tests fail because `main()` has not invoked the uploader.

- [ ] **Step 7: Invoke upload only after local config is written and wrap failures with a clear local-results message**

Immediately after `config_path.write_text(...)` in `run_probe.py`:

```python
if args.upload_repo:
    try:
        upload_result_dir(
            result_dir=result_dir,
            repo_id=args.upload_repo,
            remote_path=upload_path,
        )
    except Exception as error:
        raise RuntimeError(
            f"Experiment completed and local results are in {result_dir}, "
            f"but Hugging Face backup failed: {error}"
        ) from error
```

Keep the normal completion message after this block, so a failed requested backup never prints an overall-success completion message.

- [ ] **Step 8: Run all `run_probe` mode/upload tests**

Run:

```bash
python -m pytest -q tests/test_run_probe_modes.py
```

Expected: all tests pass, including existing vLLM spawn/CUDA-cache behavior.

- [ ] **Step 9: Run uploader and run-probe tests together**

Run:

```bash
python -m pytest -q tests/test_results_upload.py tests/test_run_probe_modes.py
```

Expected: all tests pass.

- [ ] **Step 10: Commit the `run_probe.py` integration**

```bash
git add run_probe.py tests/test_run_probe_modes.py
git commit -m "feat: upload completed probe results to Hugging Face"
```

---

### Task 3: Document RunPod/Hugging Face backup usage and verify regressions

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents the Task 2 CLI exactly: `--upload-repo <owner/repo>`.
- Documents `HF_TOKEN` as environment-only authentication.
- Documents remote layout `runs/<timestamp>-<result-dir-name>/`.

- [ ] **Step 1: Add a concise README example under the RunPod vLLM section**

Add text after the existing secret guidance. Keep the token value symbolic:

```markdown
To back up a completed run automatically, expose `HF_TOKEN` through the
RunPod environment/Secret and pass a pre-existing Hugging Face Dataset repo:

```bash
export HF_TOKEN="<provided-by-secret>"

python run_probe.py \
  --engine vllm \
  --only-rl \
  --rollouts 256 \
  --result-dir results_rl256_vllm \
  --upload-repo UnitedSnakes/rlvr-behavior-probe-results
```

The local files are written first. A successful backup is stored under a
run-start timestamped path such as:

```text
runs/20260825T235312Z-results_rl256_vllm/
```

The destination Dataset repo must already exist. If backup fails, the local
result directory is preserved and the command exits unsuccessfully.
```

Use Markdown fence nesting that renders correctly in the final README.

- [ ] **Step 2: Run the focused test suite after documentation changes**

Run:

```bash
python -m pytest -q tests/test_results_upload.py tests/test_run_probe_modes.py tests/test_vllm_sampler.py
```

Expected: all tests pass.

- [ ] **Step 3: Run the broader repository tests available in the current environment**

Run:

```bash
python -m pytest -q
```

Expected: all tests that can collect in the configured project environment pass. If environment-only dependency failures occur, record the exact failing command/output and do not claim full-suite success.

- [ ] **Step 4: Verify no credential was added to tracked text**

Run:

```bash
git grep -nE 'hf_[A-Za-z0-9]{20,}|HF_TOKEN=' -- ':!docs/superpowers/plans/*'
```

Expected: no real Hugging Face token. Documentation may mention the variable name `HF_TOKEN`, but must not contain a token value.

- [ ] **Step 5: Review the final diff for scope and formatting**

Run:

```bash
git diff --check
git diff -- README.md run_probe.py probe/results_upload.py tests/test_results_upload.py tests/test_run_probe_modes.py
```

Expected: `git diff --check` reports no whitespace errors; diff contains only the approved result-upload feature and documentation.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md
git commit -m "docs: document Hugging Face result backup"
```

---

## Final verification gate

Before claiming the feature is complete, run the verification commands from Task 3 on the actual implementation environment and inspect their fresh output. In particular, do not infer operational success from unit tests alone: the first real RunPod use should perform one small `--upload-repo` run against the intended pre-existing Dataset repo and confirm that the expected timestamped directory appears remotely while local files remain present.

A real Hub smoke test must use `HF_TOKEN` from the environment/RunPod Secret and must never print the token. Do not delete the local result directory until the remote files have been inspected.
