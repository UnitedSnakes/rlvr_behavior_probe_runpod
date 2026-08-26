from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_installs_executable_bootstrap():
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")

    assert (
        "COPY --chmod=0755 docker/rlvr-bootstrap.sh "
        "/usr/local/bin/rlvr-bootstrap"
    ) in dockerfile
    assert "RUN test -x /usr/local/bin/rlvr-bootstrap" in dockerfile


def test_normal_push_does_not_automatically_replace_stable_image_tag():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "build-runpod-image.yml"
    ).read_text(encoding="utf-8")

    assert "publish_stable:" in workflow
    assert "type=sha,prefix=sha-" in workflow
    assert (
        "type=raw,value=0.27.1,"
        "enable=${{ github.event_name == 'workflow_dispatch' && inputs.publish_stable }}"
    ) in workflow


def test_readme_documents_bootstrap_template_and_correct_hf_repo():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    required = [
        "GITHUB_DEPLOY_KEY_B64={{ RUNPOD_SECRET_github_rlvr_deploy_key_b64 }}",
        "HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}",
        "RLVR_REPO=git@github.com:UnitedSnakes/rlvr_behavior_probe_runpod.git",
        "RLVR_BRANCH=difficulty-bin-analysis",
        "RLVR_REPO_DIR=/workspace/rlvr_behavior_probe_runpod",
        "/workspace/rlvr-bootstrap.log",
        "rlvr-bootstrap",
        "HKReporter/rlvr-behavior-probe-results",
    ]
    for text in required:
        assert text in readme

    assert "UnitedSnakes/rlvr-behavior-probe-results" not in readme
