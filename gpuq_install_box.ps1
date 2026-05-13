# gpuq_install_box.ps1 -- one-time installer for the gpuq daemon on the comfy box.
#
# Creates C:\gpuq\{daemon.py,pending,running,done,logs}, then registers a
# scheduled task `GpuQueueDaemon` that runs the daemon at every user logon and
# at system boot. The task auto-restarts on failure and has no execution-time
# limit. Deploy by scp'ing this file + scripts/gpuq_daemon.py to C:\gpuq\, then
# running:
#
#     powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\gpuq\install.ps1
#
# (The Mac-side caller in scripts/gpuq.py handles all of that for you via
# `gpuq.py daemon install` -- this file is the canonical recipe.)

$ErrorActionPreference = "Stop"

$root = "C:\gpuq"
foreach ($d in @($root, "$root\pending", "$root\running", "$root\done", "$root\logs")) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}

# Use the ai-toolkit venv's python so the daemon picks up any installed deps;
# for stdlib-only the system python would also work.
$py = "C:\ai-toolkit\venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    # fall back to system python if the ai-toolkit venv isn't present
    $py = "python.exe"
}

$daemon = "$root\daemon.py"
if (-not (Test-Path $daemon)) {
    throw "missing $daemon -- scp scripts/gpuq_daemon.py to C:\gpuq\daemon.py first"
}

$action  = New-ScheduledTaskAction -Execute $py -Argument "-u `"$daemon`"" -WorkingDirectory $root
$trigger1 = New-ScheduledTaskTrigger -AtLogOn
$trigger2 = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -RestartCount 99 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive

Unregister-ScheduledTask -TaskName GpuQueueDaemon -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName GpuQueueDaemon `
    -Action $action -Trigger @($trigger1, $trigger2) -Settings $settings -Principal $principal `
    | Out-Null
Start-ScheduledTask -TaskName GpuQueueDaemon
Write-Output "GpuQueueDaemon installed and started (python: $py, daemon: $daemon)"
Write-Output "Heartbeat: $root\daemon.heartbeat"
