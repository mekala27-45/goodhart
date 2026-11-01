# goodhart GPU kernel, built from kernels/templates/run.py for one committed request.
# It downloads the request's bundle from the repository's release by URL, checks its sha256, installs
# the pinned requirements with their hashes, and runs the stage. The bundle never carries the
# held-out package, the twins or any test gold. Outputs go to /kaggle/working/out, which the gpu
# workflow downloads and checks by digest. Nothing here reads or prints a credential.
import hashlib
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

REQUEST_ID = "__REQUEST_ID__"
BUNDLE_URL = "__BUNDLE_URL__"
BUNDLE_SHA256 = "__BUNDLE_SHA256__"

WORK = Path("/tmp/goodhart")
OUT = Path("/kaggle/working/out")


def fetch_bundle() -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    target = WORK / "bundle.tar.gz"
    with urllib.request.urlopen(BUNDLE_URL, timeout=300) as response:
        data = response.read()
    found = hashlib.sha256(data).hexdigest()
    if found != BUNDLE_SHA256:
        raise SystemExit(f"bundle digest {found[:12]} does not match the request's {BUNDLE_SHA256[:12]}")
    target.write_bytes(data)
    folder = WORK / "bundle"
    with tarfile.open(target) as tar:
        tar.extractall(folder, filter="data")
    return folder


def install(folder: Path) -> None:
    req = folder / "requirements-gpu.txt"
    done = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--require-hashes", "--no-deps", "-r", str(req)],
        capture_output=True,
        text=True,
    )
    print(done.stdout[-2000:])
    if done.returncode != 0:
        print(done.stderr[-4000:], file=sys.stderr)
        raise SystemExit("pinned requirements did not install")
    # The image ships torchao 0.10, which PEFT 0.21 refuses on import of any LoRA layer; nothing here
    # uses torchao, so it is removed rather than upgraded against the image's PyTorch.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)


def main() -> int:
    folder = fetch_bundle()
    install(folder)
    sys.path.insert(0, str(folder / "src"))
    os.environ.setdefault("GOODHART_INPUT_ROOT", "/kaggle/input")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    OUT.mkdir(parents=True, exist_ok=True)
    from goodhart_train.entry import run

    return run(folder, OUT, WORK / "cache")


if __name__ == "__main__":
    code = main()
    print(f"goodhart request {REQUEST_ID} finished with status {code}")
