#!/usr/bin/env python3
"""gpuq — submit GPU jobs to a SSH-reachable Windows GPU box's queue daemon.

The daemon (``gpuq_daemon.py`` deployed to ``C:\\gpuq\\daemon.py``) polls
``C:\\gpuq\\pending\\*.json`` and runs jobs serially. This CLI is the
producer side — drops job JSONs into pending/ over SSH, queries state,
tails logs, pulls outputs back.

Built for the ai-toolkit FLUX.2 dual-GPU training + inference workflow
(see ``genno-whittlery/flux2-dual-gpu-lora``) but the daemon is just a
FIFO subprocess runner — any ``run.py <yaml>`` or ``inference/cli.py ...``
shaped command works.

Examples:

    # Submit an ai-toolkit training run (yaml already on the box):
    gpuq.py submit train --name kasumi-v2 \\
        --yaml 'C:\\ai-toolkit\\config\\kasumi-flux2-v2.yaml'

    # Submit an inference run, scp'ing the prompts JSON up first:
    gpuq.py submit infer --name kasumi-gallery \\
        --lora 'C:\\ai-toolkit\\output\\kasumi-flux2-v1\\kasumi-flux2-v1.safetensors' \\
        --prompts ./kasumi_prompts.json \\
        --out 'C:\\tmp\\kasumi-gallery' \\
        --seeds 42,43,44 --strengths 1.0

    gpuq.py gpu                # show GPU status (nvidia-smi summary)
    gpuq.py list               # also prints the GPU header
    gpuq.py status <id>
    gpuq.py tail <id>          # tail the running job's log
    gpuq.py kill <id>          # remove from pending OR terminate a running job
    gpuq.py drain              # cancel every pending job
    gpuq.py clear [--orphans]  # kill running + drain pending (+ kill non-gpuq python.exe)
    gpuq.py pull   <id> <dest> # pull infer outputs back to a local dir

Connects via the ``GPUQ_HOST`` env var (e.g. ``user@gpubox.tail1234.ts.net``).
``COMFY_HOST`` is accepted as a legacy alias.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

HOST = os.environ.get("GPUQ_HOST") or os.environ.get("COMFY_HOST")
if not HOST:
    sys.stderr.write(
        "gpuq: set GPUQ_HOST=user@hostname for the SSH-reachable Windows GPU box "
        "(e.g. GPUQ_HOST=mindos@gpubox.tail1234.ts.net). The daemon expects to "
        "live on the Windows side under C:\\gpuq\\ (WSL path /mnt/c/gpuq/).\n"
    )
    sys.exit(2)
QDIR = "/mnt/c/gpuq"
PEND = f"{QDIR}/pending"
RUN  = f"{QDIR}/running"
DONE = f"{QDIR}/done"
LOGS = f"{QDIR}/logs"

DEFAULT_ENV = {"FLUX2_DUAL_GPU": "true", "FLUX2_TE_DEVICE": "cpu"}


# ----------------------------------------------------------------- ssh helpers

def ssh(cmd: str, *, timeout: int = 30, check: bool = True) -> str:
    """Run a shell command on the box over ssh, return stdout (text).

    Tolerates Windows-side non-UTF-8 output (PowerShell, schtasks) with errors='replace'.
    """
    res = subprocess.run(
        ["ssh", "-o", f"ConnectTimeout={min(timeout, 15)}", HOST, cmd],
        capture_output=True, timeout=timeout,
    )
    stdout = res.stdout.decode("utf-8", errors="replace")
    stderr = res.stderr.decode("utf-8", errors="replace")
    if check and res.returncode != 0:
        sys.stderr.write(f"ssh failed (rc={res.returncode}): {cmd}\n{stderr}")
        sys.exit(2)
    return stdout


def scp_to(local: Path, remote: str) -> None:
    subprocess.run(["scp", "-o", "ConnectTimeout=15", str(local), f"{HOST}:{remote}"],
                   check=True)


def scp_from(remote_glob: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    # Use a single ssh+tar trick so globs expand on the remote side cleanly:
    cmd = f"cd $(dirname {shlex.quote(remote_glob)}) && tar cf - $(basename {shlex.quote(remote_glob)})"
    pipe = subprocess.Popen(["ssh", HOST, cmd], stdout=subprocess.PIPE)
    subprocess.run(["tar", "xf", "-", "-C", str(local_dir)], stdin=pipe.stdout, check=True)
    pipe.wait()


def submit_json(job: dict) -> str:
    """Write the job JSON to a temp local file, scp into pending/ on the box."""
    jid = job["id"]
    local = Path("/tmp") / f"gpuq-{jid}.json"
    local.write_text(json.dumps(job, indent=2), encoding="utf-8")
    scp_to(local, f"{PEND}/{jid}.json")
    local.unlink(missing_ok=True)
    return jid


# ---------------------------------------------------------------- box helpers

PS_EXE = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
NVIDIA_SMI = "/mnt/c/Windows/System32/nvidia-smi.exe"


def _ensure_helpers() -> None:
    """Lazy-deploy the kill_match.ps1 helper to C:\\gpuq\\ if missing."""
    here = Path(__file__).resolve().parent
    remote = "/mnt/c/gpuq/kill_match.ps1"
    out = ssh(f"test -f {remote} && echo yes || echo no", check=False).strip()
    if out != "yes":
        scp_to(here / "gpuq_kill_match.ps1", remote)


def _kill_by_substring(substring: str) -> str:
    """Kill any python.exe on the box whose CommandLine contains substring.

    Returns the PowerShell output (one line per killed PID, or "no match" line).
    """
    if "'" in substring:
        sys.exit(f"kill substring may not contain single quotes: {substring}")
    _ensure_helpers()
    cmd = (f"{PS_EXE} -NoProfile -ExecutionPolicy Bypass "
           f"-File 'C:\\gpuq\\kill_match.ps1' "
           f"-Substring '{substring}' 2>&1 | tr -d '\\r'")
    return ssh(cmd, timeout=30)


def _job_kill_substring(job: dict) -> str | None:
    """Return the unique CommandLine fragment for a job, or None if not killable."""
    t = job.get("type")
    if t == "train":
        yaml = job.get("yaml") or ""
        # The yaml basename (e.g., "sumi-flux2-v1.yaml") is unique enough.
        base = yaml.replace("/", "\\").rsplit("\\", 1)[-1]
        return base or None
    if t == "infer":
        # The --out path is unique per job.
        return job.get("out") or None
    return None


def _gpu_lines(samples: int = 5, interval_ms: int = 150) -> list[str]:
    """Return one-line-per-GPU strings from nvidia-smi, reporting *peak* util across N samples.

    A single nvidia-smi snapshot mis-reports dual-GPU pipeline-parallel work as
    single-GPU: at any instant one half of the transformer is forwarding and the
    other half is idle. Sampling 5× over ~600 ms widely catches both peaks even at
    ~2.8 s/it FLUX.2 training speed. VRAM is the truer "is this GPU in use" signal.
    """
    q = (f"{NVIDIA_SMI} --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu "
         f"--format=csv,noheader,nounits")
    sleep = f"sleep {interval_ms/1000:.3f}"
    parts = [q]
    for _ in range(samples - 1):
        parts.extend([sleep, "echo ---", q])
    raw = ssh(f"({'; '.join(parts)}) 2>&1 | tr -d '\\r'", check=False).strip()

    # Aggregate by GPU index: take peak util across samples; memory/temp from latest sample.
    by_gpu: dict[str, dict] = {}
    for line in raw.splitlines():
        if line.startswith("---"):
            continue
        cols = [p.strip() for p in line.split(",")]
        if len(cols) != 5:
            continue
        idx, util, mused, mtotal, temp = cols
        try:
            util_i = int(util)
        except ValueError:
            continue
        g = by_gpu.setdefault(idx, {"util_max": util_i})
        g["util_max"] = max(g["util_max"], util_i)
        g["mused"], g["mtotal"], g["temp"] = mused, mtotal, temp
    out = []
    for idx in sorted(by_gpu.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        g = by_gpu[idx]
        try:
            vram_pct = round(int(g["mused"]) * 100 / int(g["mtotal"]))
        except (ValueError, ZeroDivisionError):
            vram_pct = 0
        out.append(
            f"  GPU {idx}: {g['util_max']:>3}% util  "
            f"{vram_pct:>3}% VRAM ({g['mused']:>5}/{g['mtotal']:>5} MiB)  "
            f"{g['temp']:>2}°C"
        )
    return out


# ----------------------------------------------------------------- id helpers

_SAFE = re.compile(r"[^a-z0-9._-]+")


def make_id(name: str, kind: str) -> str:
    slug = _SAFE.sub("-", name.lower()).strip("-") or "job"
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:4]
    return f"{kind}-{slug}-{ts}-{short}"


def parse_env(items: list[str]) -> dict[str, str]:
    out = dict(DEFAULT_ENV)
    for kv in items or []:
        if "=" not in kv:
            sys.exit(f"--env expects KEY=VALUE, got: {kv}")
        k, v = kv.split("=", 1)
        out[k] = v
    return out


# ----------------------------------------------------------------- verbs

def cmd_submit_train(args: argparse.Namespace) -> int:
    if not args.yaml.lower().endswith((".yaml", ".yml")):
        sys.exit("--yaml should point at an ai-toolkit yaml on the box")
    jid = make_id(args.name, "train")
    job = {
        "id": jid, "name": args.name, "type": "train",
        "yaml": args.yaml, "env": parse_env(args.env),
        "submitted_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    submit_json(job)
    print(jid)
    return 0


def cmd_submit_infer(args: argparse.Namespace) -> int:
    # prompts may be a local file — scp it up under C:\tmp\
    prompts = args.prompts
    if Path(prompts).exists():
        remote = f"/mnt/c/tmp/{Path(prompts).name}"
        scp_to(Path(prompts), remote)
        prompts = remote.replace("/mnt/c/", "C:\\").replace("/", "\\")
    elif not prompts.lower().startswith("c:\\"):
        sys.exit(f"--prompts must be a local file or a Windows path on the box, got: {prompts}")

    jid = make_id(args.name, "infer")
    job = {
        "id": jid, "name": args.name, "type": "infer",
        "lora": args.lora,
        "prompts": prompts,
        "out": args.out,
        "strengths": args.strengths,
        "steps": args.steps, "cfg": args.cfg,
        "width": args.width, "height": args.height,
        "seed": args.seed,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "env": parse_env(args.env),
        "submitted_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if args.seeds:
        job["seeds"] = args.seeds
    submit_json(job)
    print(jid)
    return 0


def _list_dir(remote_dir: str) -> list[str]:
    out = ssh(f"ls -1 {remote_dir} 2>/dev/null || true").strip()
    return [l for l in out.splitlines() if l.endswith(".json")]


def cmd_list(args: argparse.Namespace) -> int:
    gpu_lines = _gpu_lines()
    if gpu_lines:
        print("# gpu")
        for ln in gpu_lines:
            print(ln)
    pendings = _list_dir(PEND)
    runnings = _list_dir(RUN)
    dones = _list_dir(DONE)
    if not args.done:
        dones = dones[-args.tail:] if args.tail else dones
    print(f"# pending ({len(pendings)})")
    for p in pendings: print(f"  {p[:-5]}")
    print(f"# running ({len(runnings)})")
    for p in runnings: print(f"  {p[:-5]}")
    print(f"# done    (showing {len(dones)})")
    # parse status from each done file (best-effort, in one ssh call)
    if dones:
        names = " ".join(shlex.quote(f"{DONE}/{n}") for n in dones)
        raw = ssh(f"for f in {names}; do "
                  "echo \"-- $(basename $f .json) --\"; "
                  "python3 -c \"import json,sys;d=json.load(open(sys.argv[1]));"
                  "print(d.get('status','?'),'rc=',d.get('exit_code','?'),"
                  "'started=',d.get('started','?'),'finished=',d.get('finished','?'))\" "
                  "$f 2>/dev/null || echo '(parse error)'; "
                  "done")
        print(raw)
    return 0


def _find(jid: str) -> tuple[str, str] | None:
    """Locate a job: returns ('pending'|'running'|'done', remote_path)."""
    for state, d in (("pending", PEND), ("running", RUN), ("done", DONE)):
        rc = ssh(f"test -f {d}/{jid}.json && echo yes || echo no").strip()
        if rc == "yes":
            return state, f"{d}/{jid}.json"
    return None


def cmd_status(args: argparse.Namespace) -> int:
    hit = _find(args.id)
    if not hit:
        print(f"not found: {args.id}", file=sys.stderr)
        return 1
    state, path = hit
    print(f"# state: {state}")
    print(ssh(f"cat {path}"))
    return 0


def cmd_tail(args: argparse.Namespace) -> int:
    log = f"{LOGS}/{args.id}.log"
    rc = ssh(f"test -f {log} && echo yes || echo no").strip()
    if rc != "yes":
        print(f"no log yet for {args.id}", file=sys.stderr)
        return 1
    n = args.lines
    print(ssh(f"tail -n {n} {log}"))
    return 0


def cmd_kill(args: argparse.Namespace) -> int:
    """Kill a pending OR running job.

    - pending: remove the JSON from pending/ (daemon never picks it up).
    - running: read the running JSON, find a unique CommandLine fragment, and
      SIGKILL the matching python.exe on the box. The daemon's subprocess.run()
      returns with a non-zero rc and rolls the job into done/ as status=failed.
    """
    hit = _find(args.id)
    if not hit:
        print(f"not found: {args.id}", file=sys.stderr)
        return 1
    state, path = hit
    if state == "pending":
        ssh(f"rm -f {path}")
        print(f"cancelled pending: {args.id}")
        return 0
    if state == "done":
        print(f"{args.id} already done — nothing to kill", file=sys.stderr)
        return 2
    # running: read job spec, find unique kill substring
    raw = ssh(f"cat {path}")
    try:
        job = json.loads(raw)
    except Exception:
        print(f"could not parse running job json", file=sys.stderr); return 3
    sub = _job_kill_substring(job)
    if not sub:
        print(f"job {args.id} has no kill-identifier (type={job.get('type')!r})",
              file=sys.stderr); return 4
    print(_kill_by_substring(sub))
    return 0


def cmd_drain(args: argparse.Namespace) -> int:
    """Cancel every pending job (does NOT affect running)."""
    pendings = _list_dir(PEND)
    if not pendings:
        print("no pending jobs")
        return 0
    if args.dry_run:
        print(f"# would cancel {len(pendings)} pending:")
        for p in pendings:
            print(f"  {p[:-5]}")
        return 0
    for p in pendings:
        ssh(f"rm -f {PEND}/{p}")
    print(f"cancelled {len(pendings)} pending")
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    """Full reset: kill running + drain pending (+ optionally kill orphan python.exe)."""
    pendings = _list_dir(PEND)
    runnings = _list_dir(RUN)
    if args.dry_run:
        print(f"# would kill {len(runnings)} running and cancel {len(pendings)} pending")
        if args.orphans:
            print("# would also kill any python.exe matching run.py or cli.py")
        return 0
    # 1. kill each running job by its unique cmdline fragment
    for rjson in runnings:
        jid = rjson[:-5]
        raw = ssh(f"cat {RUN}/{rjson}", check=False)
        try:
            job = json.loads(raw)
        except Exception:
            print(f"  skip {jid}: bad json", file=sys.stderr)
            continue
        sub = _job_kill_substring(job)
        if sub:
            print(f"# kill running {jid} (match: {sub})")
            print(_kill_by_substring(sub))
    # 2. drain pending
    for p in pendings:
        ssh(f"rm -f {PEND}/{p}")
    if pendings:
        print(f"cancelled {len(pendings)} pending")
    # 3. orphans
    if args.orphans:
        print("# killing orphan python.exe matching run.py / cli.py")
        print(_kill_by_substring("run.py"))
        print(_kill_by_substring("cli.py"))
    return 0


def cmd_gpu(args: argparse.Namespace) -> int:
    """Show GPU status from nvidia-smi on the box."""
    lines = _gpu_lines()
    if not lines:
        print("(no nvidia-smi output)", file=sys.stderr)
        return 1
    for ln in lines:
        print(ln)
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    """Pull the outputs of a (done) inference job back to a local dir.

    We look up the job's `out` field in the done JSON, then scp+tar the .png files back.
    """
    hit = _find(args.id)
    if not hit:
        print(f"not found: {args.id}", file=sys.stderr); return 1
    state, path = hit
    if state == "pending":
        print(f"{args.id} still pending — nothing to pull", file=sys.stderr); return 2
    raw = ssh(f"cat {path}")
    try:
        job = json.loads(raw)
    except Exception:
        print(f"could not parse job json", file=sys.stderr); return 3
    if job.get("type") != "infer":
        print(f"{args.id} is not an infer job (type={job.get('type')!r})", file=sys.stderr); return 4
    out = job.get("out")
    if not out:
        print(f"{args.id} has no 'out' field", file=sys.stderr); return 5
    # convert Windows path → WSL /mnt/c/...
    out_wsl = out.replace("C:\\", "/mnt/c/").replace("\\", "/")
    dest = Path(args.dest)
    print(f"pulling {out_wsl}/*.png -> {dest}")
    scp_from(f"{out_wsl}/*.png", dest)
    print(f"pulled {len(list(dest.glob('*.png')))} png files")
    return 0


def cmd_daemon_status(args: argparse.Namespace) -> int:
    heart = ssh(f"cat {QDIR}/daemon.heartbeat 2>/dev/null || echo 'no heartbeat'").strip()
    task = ssh("/mnt/c/Windows/System32/schtasks.exe /query /tn GpuQueueDaemon /fo list 2>/dev/null | tr -d '\\r' | grep -E 'Status|Last' || echo '(task not registered)'").strip()
    print(f"heartbeat: {heart}")
    print(f"task:\n{task}")
    return 0


def cmd_daemon_install(args: argparse.Namespace) -> int:
    """One-time install: scp daemon.py + install.ps1 + kill_match.ps1, run the installer."""
    here = Path(__file__).resolve().parent
    daemon = here / "gpuq_daemon.py"
    installer = here / "gpuq_install_box.ps1"
    killer = here / "gpuq_kill_match.ps1"
    for f in (daemon, installer, killer):
        if not f.exists():
            sys.exit(f"missing local file: {f}")
    ssh("mkdir -p /mnt/c/gpuq")
    scp_to(daemon, "/mnt/c/gpuq/daemon.py")
    scp_to(installer, "/mnt/c/gpuq/install.ps1")
    scp_to(killer, "/mnt/c/gpuq/kill_match.ps1")
    out = ssh(r"/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe "
              r"-NoProfile -ExecutionPolicy Bypass -File 'C:\gpuq\install.ps1' 2>&1 | tr -d '\r'")
    print(out)
    return 0


# ----------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpuq",
        description="Submit GPU jobs to the comfy box queue daemon.")
    sub = p.add_subparsers(dest="verb", required=True)

    s = sub.add_parser("submit", help="submit a job")
    sub2 = s.add_subparsers(dest="kind", required=True)

    st = sub2.add_parser("train", help="submit an ai-toolkit training run")
    st.add_argument("--name", required=True)
    st.add_argument("--yaml", required=True, help="C:\\path\\to\\config.yaml on the box")
    st.add_argument("--env", action="append", default=[],
                    help="extra env: KEY=VALUE (repeatable; FLUX2_DUAL_GPU=true and FLUX2_TE_DEVICE=cpu are default)")
    st.set_defaults(func=cmd_submit_train)

    si = sub2.add_parser("infer", help="submit an inference render")
    si.add_argument("--name", required=True)
    si.add_argument("--lora", required=True, help="C:\\path on the box")
    si.add_argument("--prompts", required=True, help="local prompts JSON (scp'd up) OR C:\\path on the box")
    si.add_argument("--out", required=True, help="C:\\out\\dir on the box")
    si.add_argument("--strengths", default="1.0")
    si.add_argument("--seeds", default=None)
    si.add_argument("--steps", type=int, default=20)
    si.add_argument("--cfg", type=float, default=4.0)
    si.add_argument("--width", type=int, default=1024)
    si.add_argument("--height", type=int, default=1024)
    si.add_argument("--seed", type=int, default=42)
    si.add_argument("--lora-rank", type=int, default=16)
    si.add_argument("--lora-alpha", type=int, default=16)
    si.add_argument("--env", action="append", default=[])
    si.set_defaults(func=cmd_submit_infer)

    l = sub.add_parser("list", help="list pending / running / done jobs")
    l.add_argument("--done", action="store_true", help="show all done jobs (not just tail)")
    l.add_argument("--tail", type=int, default=10, help="limit done to last N")
    l.set_defaults(func=cmd_list)

    for verb, fn, help_ in (("status", cmd_status, "show a job's JSON"),
                            ("kill",   cmd_kill,   "remove pending OR terminate a running job"),
                            ("cancel", cmd_kill,   "alias for kill"),
                            ("tail",   cmd_tail,   "tail a job's log")):
        x = sub.add_parser(verb, help=help_)
        x.add_argument("id")
        if verb == "tail":
            x.add_argument("-n", "--lines", type=int, default=60)
        x.set_defaults(func=fn)

    pl = sub.add_parser("pull", help="pull infer outputs to a local dir")
    pl.add_argument("id")
    pl.add_argument("dest")
    pl.set_defaults(func=cmd_pull)

    g = sub.add_parser("gpu", help="show nvidia-smi summary from the box")
    g.set_defaults(func=cmd_gpu)

    dr = sub.add_parser("drain", help="cancel every pending job (does not touch running)")
    dr.add_argument("--dry-run", action="store_true", help="show what would be cancelled and exit")
    dr.set_defaults(func=cmd_drain)

    cl = sub.add_parser("clear", help="kill running + drain pending (+ optional orphan python.exe)")
    cl.add_argument("--orphans", action="store_true",
                    help="also kill any python.exe matching run.py or cli.py (catches non-gpuq jobs)")
    cl.add_argument("--dry-run", action="store_true", help="show what would be killed and exit")
    cl.set_defaults(func=cmd_clear)

    dn = sub.add_parser("daemon", help="daemon ops")
    dnsub = dn.add_subparsers(dest="daemon_verb", required=True)
    ds = dnsub.add_parser("status", help="show daemon heartbeat + scheduled-task state")
    ds.set_defaults(func=cmd_daemon_status)
    di = dnsub.add_parser("install", help="one-time deploy + register the box-side daemon")
    di.set_defaults(func=cmd_daemon_install)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
