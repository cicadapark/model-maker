# model-maker's installer for Windows. Read it before you run it:
#   powershell -ExecutionPolicy Bypass -c "irm https://github.com/cicadapark/model-maker/releases/download/v@VERSION@/install.ps1 | iex"
# ("Bypass" lasts for that one PowerShell only: nothing is set for this PC or your account.)
# What it does: downloads one archive, checks its SHA-256 against the sum written below AND the sum published on the
# documentation page, and unpacks it into a folder of your own. It needs no administrator rights, changes nothing for
# other users, and never turns off any protection of Windows.
# This build is NOT signed: the two sums are how you know it is ours.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Version       = '@VERSION@'
$ArchiveUrl    = '@ARCHIVE_URL@'
$ArchiveSha256 = '@ARCHIVE_SHA256@'
$SumsUrl       = '@SUMS_URL@'    # the documentation page's list of sums

# Where it goes: MODEL_MAKER_HOME if you set it, else your own local programs folder.
$Home_ = if ($env:MODEL_MAKER_HOME) { $env:MODEL_MAKER_HOME } else { Join-Path $env:LOCALAPPDATA 'model-maker' }
$Top   = "model-maker-$Version"
$Into  = Join-Path $Home_ $Top
$Name  = Split-Path $ArchiveUrl -Leaf

# An existing install may hold your models and environment (tens of GB): it is never replaced or deleted here.
if (Test-Path -LiteralPath $Into) {
    throw "$Into is already there. Remove it yourself, or set MODEL_MAKER_HOME to another folder. Nothing was installed."
}

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13

# The second sum, from the documentation page: the line that names this archive. Read from the reply's bytes as UTF-8:
# a file served without a text type (a release asset is application/octet-stream) has no text in .Content.
$SumsReply = Invoke-WebRequest -UseBasicParsing $SumsUrl
$SumsText  = [Text.Encoding]::UTF8.GetString($SumsReply.RawContentStream.ToArray()).TrimStart([char]0xFEFF)
$Published = ($SumsText -split "`n" |
    Where-Object { $_ -match [regex]::Escape($Name) + '\s*$' } | ForEach-Object { ($_ -split '\s+')[0].ToLower() })
if ($Published -ne $ArchiveSha256) { throw "The documentation page's sum for $Name is not this script's. Nothing was installed." }

$Temp = Join-Path ([IO.Path]::GetTempPath()) ("model-maker-" + [guid]::NewGuid())
New-Item -ItemType Directory $Temp | Out-Null
try {
    $Archive = Join-Path $Temp $Name
    Write-Host "Downloading $Name"
    Invoke-WebRequest -UseBasicParsing $ArchiveUrl -OutFile $Archive
    $Found = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLower()
    if ($Found -ne $ArchiveSha256) { throw "$Name has the sum $Found, not $ArchiveSha256. Nothing was installed." }

    # Every entry must be inside the archive's own top folder (no '..', no drive, no absolute path) before anything
    # is unpacked.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $Zip = [IO.Compression.ZipFile]::OpenRead($Archive)
    try {
        foreach ($Entry in $Zip.Entries) {
            $Path = $Entry.FullName
            if (-not $Path.StartsWith("$Top/") -or $Path -match '(^|[\\/])\.\.([\\/]|$)' -or $Path -match ':') {
                throw "Unsafe path in the archive: $Path. Nothing was installed."
            }
        }
    } finally { $Zip.Dispose() }

    New-Item -ItemType Directory -Force $Home_ | Out-Null
    Expand-Archive -LiteralPath $Archive -DestinationPath $Home_
} finally { Remove-Item -Recurse -Force -LiteralPath $Temp }

Write-Host "model-maker $Version (an unsigned build) is in $Into"
Write-Host "Next, in a new PowerShell window:"
Write-Host "  & '$Into\model-maker.exe' setup"
