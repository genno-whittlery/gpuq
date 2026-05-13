# gpuq_kill_match.ps1 -- kill python.exe processes whose CommandLine contains a given substring.
#
# Used by `scripts/gpuq.py kill <id>` and `scripts/gpuq.py clear` to terminate a running job
# without modifying the daemon. The daemon spawned the subprocess via subprocess.run, so it does
# not track PIDs separately -- we identify by the unique per-job command-line argument (yaml path
# for train, --out path for infer). Stop-Process -Force terminates the python.exe; the daemon's
# subprocess.run() returns with non-zero rc and rolls the job into done/ as status=failed.
#
# Usage:
#     powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\gpuq\kill_match.ps1 -Substring 'sumi-flux2-v1.yaml'
#
# -Substring is matched case-sensitively as a literal (not regex) -- callers pass the exact
# command-line fragment they want to match. Multiple processes can match (uv launches a parent
# python.exe that re-execs the real one); all are killed.

param(
    [Parameter(Mandatory=$true)][string]$Substring
)

$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -and $_.CommandLine.Contains($Substring)
}

if (-not $procs) {
    Write-Output "no match for: $Substring"
    exit 0
}

foreach ($p in $procs) {
    $cl = $p.CommandLine
    if ($cl.Length -gt 140) { $cl = $cl.Substring(0, 140) + "..." }
    Write-Output ("killing PID " + $p.ProcessId + " :: " + $cl)
    try {
        Stop-Process -Id $p.ProcessId -Force
    } catch {
        Write-Output ("  fail: " + $_.Exception.Message)
    }
}
