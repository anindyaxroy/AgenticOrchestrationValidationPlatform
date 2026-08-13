<#
.SYNOPSIS
    Compresses a raw .xes file into a proper .xes.gz (real gzip stream),
    with before/after verification of the gzip magic bytes.

.USAGE
    Edit $InputPath below, or run with -InputPath:
        .\Compress-XesToGz.ps1 -InputPath "BPI_Challenge_2019.xes"
#>

param(
    [string]$InputPath = "BPI_Challenge_2019.xes"
)

# Resolve to a full path relative to the current folder
$InputPath = (Resolve-Path $InputPath -ErrorAction Stop).Path
$OutputPath = "$InputPath.gz"

function Get-MagicBytes($path) {
    $bytes = [System.IO.File]::ReadAllBytes($path)[0..1]
    return "{0:X2} {1:X2}" -f $bytes[0], $bytes[1]
}

Write-Host "=== Input file ===" -ForegroundColor Cyan
$inSize = (Get-Item $InputPath).Length
Write-Host "Path:  $InputPath"
Write-Host "Size:  $([math]::Round($inSize / 1MB, 1)) MB"
$inMagic = Get-MagicBytes $InputPath
Write-Host "Magic: $inMagic"

if ($inMagic -eq "1F 8B") {
    Write-Host ""
    Write-Host "WARNING: input already has gzip magic bytes - it may already be compressed." -ForegroundColor Yellow
    $confirm = Read-Host "Continue anyway? (y/n)"
    if ($confirm -ne "y") { exit }
}

Write-Host ""
Write-Host "=== Compressing (this will take a while for a large file) ===" -ForegroundColor Cyan
$sw = [System.Diagnostics.Stopwatch]::StartNew()

$inStream  = [System.IO.File]::OpenRead($InputPath)
$outStream = [System.IO.File]::Create($OutputPath)
$gzip      = New-Object System.IO.Compression.GZipStream($outStream, [System.IO.Compression.CompressionLevel]::Optimal)

try {
    $inStream.CopyTo($gzip)
}
finally {
    $gzip.Close()
    $outStream.Close()
    $inStream.Close()
}

$sw.Stop()
Write-Host "Done in $([math]::Round($sw.Elapsed.TotalSeconds, 1))s"

Write-Host ""
Write-Host "=== Output file ===" -ForegroundColor Cyan
$outSize = (Get-Item $OutputPath).Length
Write-Host "Path:  $OutputPath"
Write-Host "Size:  $([math]::Round($outSize / 1MB, 1)) MB"
$outMagic = Get-MagicBytes $OutputPath
Write-Host "Magic: $outMagic"

$ratio = [math]::Round((1 - ($outSize / $inSize)) * 100, 1)
Write-Host "Compression: $ratio percent reduction"

Write-Host ""
if ($outMagic -eq "1F 8B") {
    Write-Host "SUCCESS: valid gzip file produced." -ForegroundColor Green
    if ($outSize -gt 200MB) {
        Write-Host "NOTE: output is still over 200MB - the nginx client_max_body_size limit will still reject it." -ForegroundColor Yellow
    }
} else {
    Write-Host "FAILED: output does not have valid gzip magic bytes. Do not upload this file." -ForegroundColor Red
}
