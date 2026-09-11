from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_installs_executable_bootstrap():
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")

    assert (
        "COPY --chmod=0755 docker/rlvr-bootstrap.sh "
        "/usr/local/bin/rlvr-bootstrap"
    ) in dockerfile
    assert "RUN test -x /usr/local/bin/rlvr-bootstrap" in dockerfile


def test_dockerfile_installs_ninja_for_flashinfer_topk_jit():
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")

    assert "ninja-build" in dockerfile


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
