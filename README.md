# gpuq

A tiny FIFO job queue for a Windows GPU box you SSH into from elsewhere.

Built for the workflow where you have **one Windows machine with one or two GPUs** doing AI training and inference, and you want to throw jobs at it from your Mac/Linux laptop without hand-writing a `start-X.bat` and a `register_X.ps1` and `Register-ScheduledTask`-ing each one. gpuq is that ceremony reduced to a one-liner with proper queueing on top.

Companion to [`genno-whittlery/flux2-dual-gpu-lora`](https://github.com/genno-whittlery/flux2-dual-gpu-lora) (the FLUX.2 dual-GPU LoRA training patch + inference engine) — gpuq is the orchestration layer above it, but the daemon itself is just a generic subprocess runner.

## Pieces

- **Local side** (`gpuq.py`) — CLI that submits jobs, queries state, tails logs, pulls outputs back. Runs on whatever you SSH from (Mac/Linux).
- **Box side** (`gpuq_daemon.py` → `C:\gpuq\daemon.py`) — a Python loop that polls `C:\gpuq\pending\*.json` and runs one job at a time. Registered as the `GpuQueueDaemon` Windows scheduled task (auto-starts at logon + boot, restarts on failure, no execution-time limit).

Jobs are simple JSON files. The daemon picks them up serially in submission order, writes a per-job log under `C:\gpuq\logs\<id>.log`, and moves the JSON into `done/` with status + exit code annotated.

## Install

You need:

- The GPU box: Windows with WSL (for the `mkdir`/`scp`/`tail` plumbing) + an ai-toolkit-style Python venv at `C:\ai-toolkit\venv\Scripts\python.exe`. Override the toolkit path with `AITK_DIR=C:\path\to\toolkit` env var on the box if it lives elsewhere.
- Your laptop: `ssh` + `scp` configured to reach the box passwordlessly.

Set the host once:

```bash
export GPUQ_HOST=mindos@gpubox.tail1234.ts.net    # whatever your box is
```

Then one-time install:

```bash
python3 gpuq.py daemon install
```

That scp's `daemon.py`, `install.ps1`, and `kill_match.ps1` to `C:\gpuq\`, registers the `GpuQueueDaemon` scheduled task, and starts it. Verify:

```bash
python3 gpuq.py daemon status
```

You should see a recent `idle` heartbeat.

## Submit jobs

```bash
# Training — yaml already on the box:
python3 gpuq.py submit train \
    --name suzurin-v3 \
    --yaml 'C:\ai-toolkit\config\suzurin-flux2-v3.yaml'

# Inference — local prompts JSON is scp'd up automatically:
python3 gpuq.py submit infer \
    --name kasumi-gallery-v2 \
    --lora 'C:\ai-toolkit\output\kasumi-flux2-v1\kasumi-flux2-v1.safetensors' \
    --prompts ./prompts/kasumi.json \
    --out 'C:\tmp\kasumi-gallery-v2' \
    --strengths 1.0 --seeds 42,43,44
```

Each submission prints the job id; the daemon picks it up on its next poll (~15 s). Default env vars passed to every job: `FLUX2_DUAL_GPU=true`, `FLUX2_TE_DEVICE=cpu` (relevant if you're running the dual-GPU FLUX.2 patch). Add more with `--env KEY=VALUE` (repeatable).

## Query state

```bash
python3 gpuq.py gpu                 # nvidia-smi summary (peak util across 5 samples,
                                    #   plus % VRAM and °C) — dual-GPU pipeline-parallel
                                    #   training shows correctly here, not as single-GPU
python3 gpuq.py list                # GPU header + pending + running + last 10 done
python3 gpuq.py list --done         # all done
python3 gpuq.py status <id>         # full job JSON (incl. status / exit_code / log path)
python3 gpuq.py tail <id> -n 100    # tail the running job's log
python3 gpuq.py pull <id> ./local/  # pull infer outputs (.png) back
```

## Cancel and clear

```bash
python3 gpuq.py kill <id>           # cancel pending OR SIGKILL the python.exe of a running job
python3 gpuq.py drain               # cancel every pending job (does not touch running)
python3 gpuq.py drain --dry-run     # preview only

python3 gpuq.py clear               # kill all running + drain all pending
python3 gpuq.py clear --orphans     # ...and SIGKILL any python.exe matching run.py / cli.py
                                    #    (catches legacy non-gpuq inference / training)
python3 gpuq.py clear --dry-run     # preview only
```

`kill <id>` for a running job locates the `python.exe` by a unique CommandLine fragment (the YAML basename for train, the `--out` path for infer) and SIGKILLs it via `gpuq_kill_match.ps1`. The daemon's `subprocess.run()` then returns with a non-zero rc and rolls the job into `done/` as `status: failed`. `cancel <id>` is kept as an alias for `kill <id>` for backward compatibility.

`clear --orphans` is the recovery hammer for when the legacy `start-X.bat` + `register_*.ps1` pattern was running before gpuq took over — it kills any `python.exe` whose command line includes `run.py` or `cli.py`, leaving the gpuq daemon (`daemon.py`) untouched.

## Layout on the box

```
C:\gpuq\
  daemon.py            # the worker loop
  install.ps1          # idempotent installer
  kill_match.ps1       # SIGKILL helper for `kill` / `clear`
  daemon.heartbeat     # touched each idle poll — "is the daemon alive"
  pending/<id>.json    # queued, not yet started
  running/<id>.json    # currently executing
  done/<id>.json       # final status + started/finished/exit_code
  logs/<id>.log        # stdout+stderr capture for the job
```

The daemon polls every 15 s and runs jobs in strict submission order (oldest first by mtime). Job IDs are auto-generated as `<train|infer>-<slug>-<YYYYMMDD-HHMMSS>-<short-hex>`.

## What it doesn't do (v1)

- **No priorities / no out-of-order.** Strict FIFO. Want a hot job? `kill` everything ahead of it first.
- **No multi-GPU partitioning.** The daemon runs jobs serially even if your box could parallelize single-GPU work across two cards. The assumption is most jobs claim the whole box (e.g., dual-GPU FLUX.2 training).
- **No retries.** A failed job stays `done/` with `status: failed`. Resubmit manually.
- **No daemon-tracked PIDs.** Killing a running job works by CommandLine substring match, not by PID. A future revision could write `running/<id>.pid` after spawning the subprocess; until then, the substring approach is reliable because each job has a unique YAML or `--out` path.
- **No box-side health beyond a heartbeat.** If the daemon's subprocess pipes deadlock, you'll see the heartbeat stop updating but no auto-recovery beyond the scheduled task's restart-on-failure.

These are conscious v1 simplifications — the cost of running them now is small for a single-developer single-box setup.

## Why this exists

If you've ever found yourself with a folder full of `start_kasumi.bat`, `register_kasumi.ps1`, `start_mikan.bat`, `register_mikan.ps1`... you know. The ceremony is the same shape every time, just with the trainer paths swapped. gpuq is that ceremony, generalized.

## License

MIT — see [LICENSE](./LICENSE).
