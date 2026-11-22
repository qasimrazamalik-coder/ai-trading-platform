<#
.SYNOPSIS
    Registers Windows Task Scheduler jobs for the paper-trading core (current user, no admin).

.DESCRIPTION
    Creates three tasks under the "\QuantOS\" folder of Task Scheduler:

      QuantOS-CoreStep     daily     python -m quant_os_core.ops.run_paper --config <cfg> --once
      QuantOS-WeeklyReport Sundays   python -m quant_os_core.ops.report --config <cfg>
      QuantOS-FundScreen   quarterly python -m quant_os_core.shariah.fund_screen <tickers>
                                       --user-agent <ua> --config <cfg> --write
                                     (runs on the first Sunday of Jan, Apr, Jul and Oct)

    Timing: the US market closes at 16:00 New York time. Pakistan (PKT, UTC+5) has no daylight
    saving, so the close is 02:00 PKT the next day in US winter (EST) and 01:00 PKT in US summer
    (EDT). The default core time 03:00 local is after the close all year for a PKT machine. On
    a machine in another time zone pass -CoreTime yourself (at least one hour after the close).

    Secrets: the tasks run as you, with your user environment variables. Set the keys once with
    setx (APCA_API_KEY_ID, APCA_API_SECRET_KEY, optional QUANT_OS_NTFY_TOPIC and
    QUANT_OS_WEBHOOK_URL). This script only checks that they exist; it never reads their
    values into a task, a file or the console.

    The tasks use the "Interactive" logon: they run while you are logged on (no password is
    stored, no admin rights are needed). Missed runs start when the PC is next available.

    Paper trading only. Nothing here touches the live-trading interlock.

.EXAMPLE
    .\scripts\setup_schedule.ps1 -Config examples\paper_core.json -UserAgent "Your Name you@example.com" -WhatIf
.EXAMPLE
    .\scripts\setup_schedule.ps1 -Config paper_core.json -UserAgent "Your Name you@example.com"
.EXAMPLE
    .\scripts\setup_schedule.ps1 -Remove
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    # Paper config (JSON), relative to the repo root or absolute.
    [string]$Config = "paper_core.json",
    # Daily core step, local time HH:mm (default 03:00: after the US close in PKT, all year).
    [ValidatePattern('^([01]\d|2[0-3]):[0-5]\d$')]
    [string]$CoreTime = "03:00",
    # Weekly report, Sunday, local time HH:mm.
    [ValidatePattern('^([01]\d|2[0-3]):[0-5]\d$')]
    [string]$ReportTime = "10:00",
    # Quarterly fund re-screen, local time HH:mm (first Sunday of Jan/Apr/Jul/Oct).
    [ValidatePattern('^([01]\d|2[0-3]):[0-5]\d$')]
    [string]$ScreenTime = "11:00",
    # SEC requires a user agent with a name and an email for EDGAR requests.
    [string]$UserAgent = "",
    # ETF tickers to re-screen; default: every "etf" row of the config's universe.
    [string[]]$FundTickers = @(),
    # Python interpreter; default: the repo's .venv.
    [string]$Python = "",
    # Remove the three tasks instead of registering them.
    [switch]$Remove
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$TaskPath = "\QuantOS\"
$Names = @("QuantOS-CoreStep", "QuantOS-WeeklyReport", "QuantOS-FundScreen")

if ($Remove) {
    foreach ($name in $Names) {
        $task = Get-ScheduledTask -TaskPath $TaskPath -TaskName $name -ErrorAction SilentlyContinue
        if ($null -eq $task) {
            Write-Host "Not registered: $name"
            continue
        }
        if ($PSCmdlet.ShouldProcess("$TaskPath$name", "Unregister scheduled task")) {
            Unregister-ScheduledTask -TaskPath $TaskPath -TaskName $name -Confirm:$false
            Write-Host "Removed: $name"
        }
    }
    return
}

if (-not $Python) { $Python = Join-Path $RepoRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path $Python)) { throw "Python not found: $Python (create the .venv or pass -Python)" }
$ConfigPath = if ([System.IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $RepoRoot $Config }
if (-not (Test-Path $ConfigPath)) { throw "Config not found: $ConfigPath" }
$ConfigPath = (Resolve-Path $ConfigPath).Path

# Keys must be user-level environment variables; only their presence is checked.
foreach ($var in @("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")) {
    if (-not [Environment]::GetEnvironmentVariable($var, "User")) {
        Write-Warning "$var is not set for your user. Run: setx $var <paper value>  (then log off and on)"
    }
}
if (-not [Environment]::GetEnvironmentVariable("QUANT_OS_NTFY_TOPIC", "User") -and
    -not [Environment]::GetEnvironmentVariable("QUANT_OS_WEBHOOK_URL", "User")) {
    Write-Warning "No QUANT_OS_NTFY_TOPIC or QUANT_OS_WEBHOOK_URL user variable: alerts use the config file or the state log only."
}

function Quote-Arg([string]$Value) {
    if ($Value -match '[\s"]') { return '"' + ($Value -replace '"', '\"') + '"' }
    return $Value
}

function New-QuantTask {
    param([string]$Name, [string]$Description, $Trigger, [string]$Execute, [string]$Arguments)
    $action = New-ScheduledTaskAction -Execute $Execute -Argument $Arguments -WorkingDirectory $RepoRoot
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 2) -MultipleInstances IgnoreNew
    Write-Host ""
    Write-Host "$Name : $Description"
    Write-Host "  $Execute $Arguments"
    if ($PSCmdlet.ShouldProcess("$TaskPath$Name", "Register scheduled task")) {
        Register-ScheduledTask -TaskPath $TaskPath -TaskName $Name -Description $Description `
            -Action $action -Trigger $Trigger -Principal $principal -Settings $settings -Force | Out-Null
        Write-Host "  registered"
    }
}

$py = Quote-Arg $Python
$cfg = Quote-Arg $ConfigPath

# 1. Daily core step after the US close.
Write-Host "NOTE: US close 16:00 New York = 02:00 PKT next day (EST) or 01:00 PKT (EDT)."
Write-Host "      Core step runs daily at $CoreTime local time; change it with -CoreTime HH:mm."
New-QuantTask -Name $Names[0] `
    -Description "Diversified core: one paper step after the US close (paper trading only)" `
    -Trigger (New-ScheduledTaskTrigger -Daily -At $CoreTime) `
    -Execute $Python -Arguments "-m quant_os_core.ops.run_paper --config $cfg --once"

# 2. Weekly report on Sunday.
New-QuantTask -Name $Names[1] `
    -Description "Weekly paper report through ntfy/webhook alerts" `
    -Trigger (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At $ReportTime) `
    -Execute $Python -Arguments "-m quant_os_core.ops.report --config $cfg"

# 3. Quarterly fund re-screen: weekly trigger, runs only on the first Sunday of a quarter.
if (-not $UserAgent) {
    Write-Warning "No -UserAgent (`"Your Name you@example.com`"): the quarterly fund re-screen is not registered."
} else {
    if ($FundTickers.Count -eq 0) {
        $json = Get-Content -Raw -Path $ConfigPath | ConvertFrom-Json
        $FundTickers = @($json.universe | Where-Object { $_.asset_class -eq "etf" } | ForEach-Object { $_.symbol })
    }
    if ($FundTickers.Count -eq 0) {
        Write-Warning "No ETF tickers in the config: the quarterly fund re-screen is not registered."
    } else {
        # PowerShell single-quoted literals: a ' inside is written as ''
        function Lit([string]$Value) { return "'" + ($Value -replace "'", "''") + "'" }
        $tickers = ($FundTickers | ForEach-Object { Lit ([string]$_) }) -join " "
        $screen = "& $(Lit $Python) -m quant_os_core.shariah.fund_screen $tickers " +
            "--user-agent $(Lit $UserAgent) --config $(Lit $ConfigPath) --write"
        $guard = "`$d = Get-Date; if (`$d.Day -le 7 -and (`$d.Month % 3) -eq 1) { $screen; exit `$LASTEXITCODE }"
        $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($guard))
        New-QuantTask -Name $Names[2] `
            -Description "Quarterly Shariah fund re-screen from SEC filings (writes financials to the config)" `
            -Trigger (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At $ScreenTime) `
            -Execute "powershell.exe" `
            -Arguments "-NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand $encoded"
        Write-Host "  (decoded: $guard)"
    }
}

Write-Host ""
Write-Host "Done. Check with: Get-ScheduledTask -TaskPath '$TaskPath'   Remove with: -Remove"
