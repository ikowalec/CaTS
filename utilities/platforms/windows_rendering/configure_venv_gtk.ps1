[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$VenvPath = ".venv",

    [Parameter(Mandatory = $false)]
    [string]$GtkRoot = "C:\Program Files\GTK2-Runtime Win64"
)

$ErrorActionPreference = "Stop"

$ps1MarkerStart = "# REACTION_HUB GTK PATH START"
$ps1MarkerEnd = "# REACTION_HUB GTK PATH END"
$ps1LegacyMarkerStart = "# >>> REACTION_HUB GTK PATH >>>"
$ps1LegacyMarkerEnd = "# <<< REACTION_HUB GTK PATH <<<"

$batMarkerStart = "REM REACTION_HUB_GTK_PATH_START"
$batMarkerEnd = "REM REACTION_HUB_GTK_PATH_END"
$batLegacyMarkerStart = "REM >>> REACTION_HUB GTK PATH >>>"
$batLegacyMarkerEnd = "REM <<< REACTION_HUB GTK PATH <<<"

function Get-Newline {
    param([string]$Text)
    if ($Text.Contains("`r`n")) {
        return "`r`n"
    }
    return "`n"
}

function Read-TextWithEncoding {
    param([string]$Path)

    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        $encoding = New-Object System.Text.UTF8Encoding($true)
        $offset = 3
    }
    elseif ($bytes.Length -ge 2 -and $bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE) {
        $encoding = New-Object System.Text.UnicodeEncoding($false, $true)
        $offset = 2
    }
    elseif ($bytes.Length -ge 2 -and $bytes[0] -eq 0xFE -and $bytes[1] -eq 0xFF) {
        $encoding = New-Object System.Text.UnicodeEncoding($true, $true)
        $offset = 2
    }
    else {
        $encoding = New-Object System.Text.UTF8Encoding($false)
        $offset = 0
    }

    $text = $encoding.GetString($bytes, $offset, $bytes.Length - $offset)
    return @{
        Text = $text
        Encoding = $encoding
    }
}

function Remove-MarkedBlock {
    param(
        [string]$Text,
        [string]$StartMarker,
        [string]$EndMarker
    )

    $pattern = "(?s)$([Regex]::Escape($StartMarker)).*?$([Regex]::Escape($EndMarker))(?:\r?\n)?"
    return [Regex]::Replace($Text, $pattern, "")
}

function Remove-MarkedBlocks {
    param(
        [string]$Text,
        [hashtable[]]$MarkerPairs
    )

    $result = $Text
    foreach ($markerPair in $MarkerPairs) {
        $result = Remove-MarkedBlock -Text $result -StartMarker $markerPair.Start -EndMarker $markerPair.End
    }
    return $result
}

function Insert-BlockAfterAnchor {
    param(
        [string]$Text,
        [string]$Anchor,
        [string]$Block,
        [string]$Newline
    )

    $anchorIndex = $Text.IndexOf($Anchor)
    if ($anchorIndex -lt 0) {
        throw "Anchor line not found: $Anchor"
    }

    $insertAt = $anchorIndex + $Anchor.Length
    $prefix = $Text.Substring(0, $insertAt)
    $suffix = $Text.Substring($insertAt)
    return $prefix + $Newline + $Block + $suffix
}

function Update-ActivationFile {
    param(
        [string]$Path,
        [string]$Anchor,
        [hashtable[]]$MarkerPairs,
        [string[]]$BlockLines
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Warning "Activation file not found, skipping: $Path"
        return $false
    }

    $inputFile = Read-TextWithEncoding -Path $Path
    $original = $inputFile.Text
    $encoding = $inputFile.Encoding
    $outputEncoding = $encoding
    if ($Path.ToLowerInvariant().EndsWith(".bat") -or $Path.ToLowerInvariant().EndsWith(".ps1")) {
        # Force UTF-8 without BOM to avoid cmd.exe / PowerShell issues with UTF-16 files.
        $outputEncoding = New-Object System.Text.UTF8Encoding($false)
    }
    $newline = Get-Newline -Text $original
    $withoutOldBlocks = Remove-MarkedBlocks -Text $original -MarkerPairs $MarkerPairs
    $block = ($BlockLines -join $newline) + $newline
    $updated = Insert-BlockAfterAnchor -Text $withoutOldBlocks -Anchor $Anchor -Block $block -Newline $newline

    if ($updated -ne $original) {
        [System.IO.File]::WriteAllText($Path, $updated, $outputEncoding)
        Write-Host "Updated $Path"
        return $true
    }

    Write-Host "No changes needed for $Path"
    return $false
}

if (-not (Test-Path -LiteralPath $VenvPath)) {
    throw "Venv path not found: $VenvPath"
}

$resolvedVenvPath = (Resolve-Path -LiteralPath $VenvPath).Path
$scriptsDir = Join-Path $resolvedVenvPath "Scripts"
$activatePs1 = Join-Path $scriptsDir "Activate.ps1"
$activateBat = Join-Path $scriptsDir "activate.bat"

$gtkRootEscapedPs1 = $GtkRoot.Replace("'", "''")
$ps1BlockTemplate = @'
# REACTION_HUB GTK PATH START
$env:GTK_RUNTIME_ROOT = '__GTK_ROOT__'
$gtkBin = Join-Path $env:GTK_RUNTIME_ROOT 'bin'
$gtkLib = Join-Path $env:GTK_RUNTIME_ROOT 'lib'
$gtkPathEntries = @($gtkBin, $gtkLib) | Where-Object { Test-Path -LiteralPath $_ }
foreach ($entry in $gtkPathEntries) {
    if (-not ($env:PATH.Split([System.IO.Path]::PathSeparator) -contains $entry)) {
        $env:PATH = "$($env:PATH)$([System.IO.Path]::PathSeparator)$entry"
    }
}
if ($gtkPathEntries.Count -gt 0) {
    $env:CAIROCFFI_DLL_DIRECTORIES = ($gtkPathEntries -join [System.IO.Path]::PathSeparator)
}
# REACTION_HUB GTK PATH END
'@
$ps1Block = ($ps1BlockTemplate.TrimEnd("`r", "`n").Replace("__GTK_ROOT__", $gtkRootEscapedPs1) -split "\r?\n")

$batBlockTemplate = @'
REM REACTION_HUB_GTK_PATH_START
set "GTK_RUNTIME_ROOT=__GTK_ROOT__"
set "GTK_RUNTIME_BIN=%GTK_RUNTIME_ROOT%\bin"
set "GTK_RUNTIME_LIB=%GTK_RUNTIME_ROOT%\lib"
if exist "%GTK_RUNTIME_BIN%" set "PATH=%PATH%;%GTK_RUNTIME_BIN%"
if exist "%GTK_RUNTIME_LIB%" set "PATH=%PATH%;%GTK_RUNTIME_LIB%"
if exist "%GTK_RUNTIME_BIN%" (
    if exist "%GTK_RUNTIME_LIB%" (
        set "CAIROCFFI_DLL_DIRECTORIES=%GTK_RUNTIME_BIN%;%GTK_RUNTIME_LIB%"
    ) else (
        set "CAIROCFFI_DLL_DIRECTORIES=%GTK_RUNTIME_BIN%"
    )
) else (
    if exist "%GTK_RUNTIME_LIB%" set "CAIROCFFI_DLL_DIRECTORIES=%GTK_RUNTIME_LIB%"
)
REM REACTION_HUB_GTK_PATH_END
'@
$batBlock = ($batBlockTemplate.TrimEnd("`r", "`n").Replace("__GTK_ROOT__", $GtkRoot) -split "\r?\n")

$changedPs1 = Update-ActivationFile `
    -Path $activatePs1 `
    -Anchor '$Env:PATH = "$VenvExecDir$([System.IO.Path]::PathSeparator)$Env:PATH"' `
    -MarkerPairs @(
        @{ Start = $ps1MarkerStart; End = $ps1MarkerEnd },
        @{ Start = $ps1LegacyMarkerStart; End = $ps1LegacyMarkerEnd }
    ) `
    -BlockLines $ps1Block

$changedBat = Update-ActivationFile `
    -Path $activateBat `
    -Anchor 'set PATH=%VIRTUAL_ENV%\Scripts;%PATH%' `
    -MarkerPairs @(
        @{ Start = $batMarkerStart; End = $batMarkerEnd },
        @{ Start = $batLegacyMarkerStart; End = $batLegacyMarkerEnd }
    ) `
    -BlockLines $batBlock

Write-Host ""
if ($changedPs1 -or $changedBat) {
    Write-Host "GTK runtime paths are now configured in the venv activation scripts."
    Write-Host "Deactivate and reactivate the environment to apply the changes."
}
else {
    Write-Host "No modifications were required."
}
