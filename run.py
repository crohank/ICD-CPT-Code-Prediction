"""
One-command launcher for the ICD-10 prediction app.

Usage:
    python run.py                # start API + Streamlit demo
    python run.py --api-only     # start only the FastAPI backend
    python run.py --demo-only    # start only the Streamlit frontend
    python run.py --install      # pip-install all requirements first, then start
    python run.py --check        # verify model artifacts + deps, don't start anything
    python run.py --port 8000 --demo-port 8501

What it does:
    1. (optional) installs requirements*.txt
    2. checks that required model artifacts exist on disk
    3. launches `uvicorn api.app:app` as a subprocess
    4. waits for GET /health to return 200
    5. launches `streamlit run demo/streamlit_app.py` as a subprocess
    6. forwards Ctrl+C to both children and shuts down cleanly

No external dependencies beyond the Python stdlib.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ── Required artifacts ─────────────────────────────────────────────────
REQUIRED_FILES = [
    ROOT / "datasets" / "processed" / "mlb.pkl",
    ROOT / "datasets" / "processed" / "tfidf_vectorizer.pkl",
    ROOT / "data" / "models" / "model_a" / "clf_sgd.pkl",
    ROOT / "data" / "models" / "model_a" / "results.json",
]

# Model C: accept either v2 or v1
MODEL_C_CANDIDATES = [
    ROOT / "data" / "models" / "model_c" / "v2" / "best_model.pt",
    ROOT / "data" / "models" / "model_c" / "best_model.pt",
]

REQUIREMENTS_FILES = [
    "requirements-dev.txt",
    "requirements-api.txt",
    "requirements-demo.txt",
]


# ── Helpers ────────────────────────────────────────────────────────────
def log(msg: str, *, prefix: str = "[run]") -> None:
    print(f"{prefix} {msg}", flush=True)


def check_artifacts() -> bool:
    """Return True if every required artifact exists on disk."""
    ok = True
    log("Checking model artifacts...")
    for f in REQUIRED_FILES:
        if f.exists():
            log(f"  OK   {f.relative_to(ROOT)}")
        else:
            log(f"  MISS {f.relative_to(ROOT)}", prefix="[run!]")
            ok = False

    if any(c.exists() for c in MODEL_C_CANDIDATES):
        found = next(c for c in MODEL_C_CANDIDATES if c.exists())
        log(f"  OK   {found.relative_to(ROOT)} (Model C)")
    else:
        log("  MISS Model C weights (looked in v2/ and root)", prefix="[run!]")
        ok = False

    return ok


def install_requirements() -> None:
    """pip install all requirements*.txt in order."""
    log("Installing requirements...")
    for r in REQUIREMENTS_FILES:
        path = ROOT / r
        if not path.exists():
            log(f"  skip {r} (not found)")
            continue
        log(f"  pip install -r {r}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(path)]
        )


def wait_for_api(port: int, timeout: float = 120.0) -> bool:
    """Poll GET /health until it returns 200 or timeout elapses."""
    url = f"http://localhost:{port}/health"
    log(f"Waiting for API at {url} (up to {int(timeout)}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    log("API is up.")
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1.0)
    return False


def start_api(port: int) -> subprocess.Popen:
    log(f"Starting FastAPI on port {port}...")
    return subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "api.app:app",
            "--host", "0.0.0.0",
            "--port", str(port),
        ],
        cwd=str(ROOT),
    )


def start_demo(port: int, api_port: int) -> subprocess.Popen:
    log(f"Starting Streamlit on port {port}...")
    env = os.environ.copy()
    env["API_URL"] = f"http://localhost:{api_port}"  # for forward-compat
    return subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            str(ROOT / "demo" / "streamlit_app.py"),
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
        ],
        cwd=str(ROOT),
        env=env,
    )


def shutdown(procs: list[subprocess.Popen]) -> None:
    log("Shutting down...")
    for p in procs:
        if p.poll() is None:
            try:
                if os.name == "nt":
                    p.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                else:
                    p.terminate()
            except Exception:
                pass
    # Grace period, then kill
    deadline = time.time() + 5.0
    for p in procs:
        remaining = max(0.1, deadline - time.time())
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            log(f"  killing pid={p.pid}")
            p.kill()


# ── Main ───────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Launch the ICD-10 prediction app.")
    ap.add_argument("--api-only", action="store_true", help="Start only the API.")
    ap.add_argument("--demo-only", action="store_true", help="Start only Streamlit.")
    ap.add_argument("--install", action="store_true",
                    help="pip-install requirements before launching.")
    ap.add_argument("--check", action="store_true",
                    help="Verify artifacts + deps and exit (don't launch).")
    ap.add_argument("--port", type=int, default=8000, help="API port (default 8000).")
    ap.add_argument("--demo-port", type=int, default=8501,
                    help="Streamlit port (default 8501).")
    ap.add_argument("--skip-artifact-check", action="store_true",
                    help="Don't bail out if model files are missing.")
    args = ap.parse_args()

    if args.install:
        install_requirements()

    artifacts_ok = check_artifacts()
    if not artifacts_ok and not args.skip_artifact_check and not args.demo_only:
        log("Missing model artifacts. Re-run with --skip-artifact-check to ignore,",
            prefix="[run!]")
        log("or train the models via notebooks_local/03,04,06,08 first.",
            prefix="[run!]")
        return 1

    if args.check:
        log("Check complete.")
        return 0 if artifacts_ok else 1

    procs: list[subprocess.Popen] = []
    api_proc: subprocess.Popen | None = None
    demo_proc: subprocess.Popen | None = None

    try:
        if not args.demo_only:
            api_proc = start_api(args.port)
            procs.append(api_proc)
            if not wait_for_api(args.port):
                log("API failed to come up in time. Aborting.", prefix="[run!]")
                shutdown(procs)
                return 2

        if not args.api_only:
            demo_proc = start_demo(args.demo_port, args.port)
            procs.append(demo_proc)
            log(f"Demo: http://localhost:{args.demo_port}")

        if not args.demo_only:
            log(f"API:  http://localhost:{args.port}  (docs at /docs)")

        log("Press Ctrl+C to stop.")

        # Block on whichever children we started; exit if any dies.
        while True:
            for p in procs:
                rc = p.poll()
                if rc is not None:
                    log(f"Child pid={p.pid} exited with code {rc}.", prefix="[run!]")
                    return rc if rc != 0 else 0
            time.sleep(1.0)

    except KeyboardInterrupt:
        log("Ctrl+C received.")
        return 0
    finally:
        shutdown(procs)


if __name__ == "__main__":
    sys.exit(main())
