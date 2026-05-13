#!/usr/bin/env python3
"""gpuq daemon — single-GPU job runner for the comfy box.

Polls ``C:\\gpuq\\pending\\*.json`` for job files, runs each to completion
(serially — the box has one shared GPU pair), writes a per-job log under
``C:\\gpuq\\logs\\``, and moves the job JSON into ``done/`` with the final
status. Runs as a persistent scheduled task.

Job JSON schema (see scripts/gpuq.py for the producer side):

    {
      "id": "<unique>",                  # used for filenames
      "name": "<human label>",
      "type": "train" | "infer",
      "submitted_at": "...",

      # train:
      "yaml": "C:\\\\ai-toolkit\\\\config\\\\...yaml",

      # infer:
      "lora":    "C:\\\\ai-toolkit\\\\output\\\\.../X.safetensors",
      "prompts": "C:\\\\tmp\\\\X_prompts.json",
      "out":     "C:\\\\tmp\\\\X-render",
      "strengths": "1.0",                # CSV
      "seeds":     "42,43,44",           # CSV, optional
      "lora_rank": 16, "lora_alpha": 16,
      "steps": 20, "cfg": 4.0, "width": 1024, "height": 1024, "seed": 42,

      "env": {"FLUX2_DUAL_GPU": "true", "FLUX2_TE_DEVICE": "cpu"}
    }

On completion the daemon writes back:
    "status": "ok" | "failed",
    "exit_code": <int>,
    "started":  "<iso>",
    "finished": "<iso>",
    "log":      "C:\\\\gpuq\\\\logs\\\\<id>.log"
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

QDIR = Path(r"C:\gpuq")
PEND = QDIR / "pending"
RUN  = QDIR / "running"
DONE = QDIR / "done"
LOGS = QDIR / "logs"
HEART = QDIR / "daemon.heartbeat"

AITK = Path(os.environ.get("AITK_DIR", r"C:\ai-toolkit"))
PY = AITK / "venv" / "Scripts" / "python.exe"
RUN_PY = AITK / "run.py"
INFER_CLI = AITK / "inference" / "cli.py"

POLL_SEC = 15


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _log(msg: str) -> None:
    print(f"[gpuq-daemon {_now()}] {msg}", flush=True)


def _ensure_dirs() -> None:
    for d in (QDIR, PEND, RUN, DONE, LOGS):
        d.mkdir(parents=True, exist_ok=True)


def _next_job() -> Path | None:
    files = sorted(PEND.glob("*.json"))
    return files[0] if files else None


def _build_cmd(job: dict) -> list[str]:
    t = job.get("type")
    if t == "train":
        yaml = job["yaml"]
        return [str(PY), "-u", str(RUN_PY), yaml]
    if t == "infer":
        cmd = [
            str(PY), "-u", str(INFER_CLI),
            "--lora", job["lora"],
            "--prompts", job["prompts"],
            "--out", job["out"],
            "--strengths", str(job.get("strengths", "1.0")),
            "--steps", str(job.get("steps", 20)),
            "--cfg", str(job.get("cfg", 4.0)),
            "--width", str(job.get("width", 1024)),
            "--height", str(job.get("height", 1024)),
            "--seed", str(job.get("seed", 42)),
            "--lora-rank", str(job.get("lora_rank", 16)),
            "--lora-alpha", str(job.get("lora_alpha", 16)),
        ]
        if job.get("seeds"):
            cmd += ["--seeds", str(job["seeds"])]
        return cmd
    raise ValueError(f"unknown job type: {t!r}")


def _run_one(job_path: Path) -> None:
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
    except Exception as e:
        # malformed — quarantine and continue
        bad = (DONE / f"{job_path.stem}.malformed.json")
        bad.write_text(f'{{"error": "malformed json: {e!r}", "raw_path": "{job_path}"}}',
                       encoding="utf-8")
        job_path.unlink(missing_ok=True)
        _log(f"malformed job moved to {bad}")
        return

    jid = job.get("id") or job_path.stem
    log_path = LOGS / f"{jid}.log"
    run_marker = RUN / job_path.name
    shutil.move(str(job_path), str(run_marker))
    _log(f"start {jid} ({job.get('type')})")

    env = os.environ.copy()
    env.update(job.get("env", {}))

    try:
        cmd = _build_cmd(job)
    except Exception as e:
        job.update(status="failed", exit_code=-1, error=str(e),
                   started=_now(), finished=_now(), log=str(log_path))
        log_path.write_text(f"failed to build cmd: {e}\n", encoding="utf-8")
        shutil.move(str(run_marker), str(DONE / run_marker.name))
        (DONE / f"{jid}.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
        return

    started = _now()
    rc = -1
    try:
        with open(log_path, "wb") as lf:
            lf.write(f"# gpuq-daemon: starting {jid} at {started}\n".encode())
            lf.write(f"# cmd: {' '.join(cmd)}\n".encode())
            lf.write(f"# cwd: {AITK}\n".encode())
            lf.write(f"# env overrides: {json.dumps(job.get('env', {}))}\n\n".encode())
            lf.flush()
            proc = subprocess.run(cmd, cwd=str(AITK), env=env,
                                  stdout=lf, stderr=subprocess.STDOUT)
            rc = proc.returncode
    except Exception as e:
        _log(f"job {jid} raised {e!r}")
        with open(log_path, "ab") as lf:
            lf.write(f"\n# gpuq-daemon: exception running cmd: {e!r}\n".encode())

    finished = _now()
    job.update(status="ok" if rc == 0 else "failed",
               exit_code=rc, started=started, finished=finished,
               log=str(log_path))
    (DONE / f"{jid}.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    run_marker.unlink(missing_ok=True)
    _log(f"done  {jid} status={job['status']} rc={rc}")


def main() -> int:
    _ensure_dirs()
    _log(f"daemon up; polling {PEND} every {POLL_SEC}s")
    HEART.write_text(f"started {_now()} pid {os.getpid()}\n", encoding="utf-8")
    try:
        while True:
            j = _next_job()
            if j:
                _run_one(j)
            else:
                HEART.write_text(f"idle {_now()} pid {os.getpid()}\n", encoding="utf-8")
                time.sleep(POLL_SEC)
    except KeyboardInterrupt:
        _log("interrupted")
        return 0


if __name__ == "__main__":
    sys.exit(main())
