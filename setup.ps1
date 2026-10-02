$ErrorActionPreference = 'Stop'

$projectRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$toolsDir = Join-Path $projectRoot 'tools'
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('ppt-takeaway-setup-' + [guid]::NewGuid().ToString('N'))
$uvVersion = '0.12.20'
$tesseractVersion = '5.5.3'
$appTessdata = Join-Path $projectRoot 'tessdata'

function Write-Step([string] $Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Download-File([string] $Uri, [string] $Destination) {
    Write-Host "Downloading $Uri"
    $null = Invoke-WebRequest -Uri $Uri -OutFile $Destination -UseBasicParsing
    if (-not (Test-Path -LiteralPath $Destination) -or (Get-Item -LiteralPath $Destination).Length -le 0) {
        throw "Download failed: $Uri"
    }
}

function Invoke-Checked([string] $Executable, [string[]] $ArgumentList, [string] $Name) {
    & $Executable @ArgumentList
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode"
    }
}

function Test-Tesseract([string] $Executable, [string] $DataDirectory) {
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { return $false }
    if (-not (Test-Path -LiteralPath (Join-Path $DataDirectory 'chi_sim.traineddata') -PathType Leaf)) { return $false }

    $oldPrefix = $env:TESSDATA_PREFIX
    try {
        $env:TESSDATA_PREFIX = $DataDirectory
        $output = & $Executable --list-langs --tessdata-dir $DataDirectory 2>&1 | Out-String
        return ($LASTEXITCODE -eq 0 -and $output -match '(?m)^\s*chi_sim\s*$')
    } catch {
        return $false
    } finally {
        $env:TESSDATA_PREFIX = $oldPrefix
    }
}

function Find-WorkingTesseract([string] $SearchRoot, [string] $ProjectData) {
    if (Test-Path -LiteralPath $SearchRoot -PathType Container) {
        $local = Get-ChildItem -LiteralPath $SearchRoot -Filter 'tesseract.exe' -File -Recurse -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($local -and (Test-Tesseract $local.FullName $ProjectData)) { return $local.FullName }
    }

    $system = Get-Command -Name 'tesseract.exe', 'tesseract' -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($system -and (Test-Tesseract $system.Source $ProjectData)) { return $system.Source }
    return $null
}

try {
    if (-not [Environment]::Is64BitOperatingSystem) {
        throw 'This setup currently supports 64-bit Windows only.'
    }

    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $null = New-Item -ItemType Directory -Path $toolsDir -Force
    $null = New-Item -ItemType Directory -Path $tempRoot -Force

    Write-Step 'Preparing a private Python runtime and application dependencies'
    $uvDirectory = Join-Path $toolsDir 'uv'
    $uvExecutable = Join-Path $uvDirectory 'uv.exe'
    if (-not (Test-Path -LiteralPath $uvExecutable -PathType Leaf)) {
        $uvArchive = Join-Path $tempRoot "uv-$uvVersion-windows-x64.zip"
        $uvExtracted = Join-Path $tempRoot 'uv-extracted'
        $uvUrl = "https://github.com/astral-sh/uv/releases/download/$uvVersion/uv-x86_64-pc-windows-msvc.zip"
        Download-File $uvUrl $uvArchive
        Expand-Archive -LiteralPath $uvArchive -DestinationPath $uvExtracted -Force
        $uvFromArchive = Get-ChildItem -LiteralPath $uvExtracted -Filter 'uv.exe' -File -Recurse |
            Select-Object -First 1
        if (-not $uvFromArchive) { throw 'The uv download did not contain uv.exe.' }
        $null = New-Item -ItemType Directory -Path $uvDirectory -Force
        Copy-Item -LiteralPath $uvFromArchive.FullName -Destination $uvExecutable -Force
    }

    $env:UV_CACHE_DIR = Join-Path $projectRoot '.uv-cache'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.python-runtime'
    $env:UV_PYTHON_INSTALL_BIN = '0'
    $env:UV_PYTHON_INSTALL_REGISTRY = '0'
    $env:UV_PYTHON_PREFERENCE = 'managed'
    Invoke-Checked -Executable $uvExecutable -ArgumentList @('python', 'install', '3.12') -Name 'Python setup'
    Invoke-Checked -Executable $uvExecutable -ArgumentList @('sync', '--project', $projectRoot, '--python', '3.12', '--frozen') -Name 'Application dependency setup'

    Write-Step 'Checking Chinese OCR support'
    $workingTesseract = Find-WorkingTesseract $toolsDir $appTessdata
    if (-not $workingTesseract) {
        Write-Host 'The portable OCR runtime will download about 200 MB; please keep this window open.'
        $tesseractPrefix = Join-Path $toolsDir 'Tesseract-OCR'
        if (Test-Path -LiteralPath $tesseractPrefix) {
            $tesseractPrefix = Join-Path $toolsDir "Tesseract-OCR-$tesseractVersion"
        }
        if (Test-Path -LiteralPath $tesseractPrefix) {
            throw "An incomplete OCR folder already exists: $tesseractPrefix"
        }

        $micromambaExecutable = Join-Path $toolsDir 'micromamba.exe'
        if (-not (Test-Path -LiteralPath $micromambaExecutable -PathType Leaf)) {
            $archive = Join-Path $tempRoot 'micromamba-win-64.tar.bz2'
            $extractDirectory = Join-Path $tempRoot 'micromamba-extracted'
            $null = New-Item -ItemType Directory -Path $extractDirectory -Force
            Download-File 'https://micro.mamba.pm/api/micromamba/win-64/latest' $archive
            $tar = Get-Command -Name 'tar.exe' -CommandType Application -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if (-not $tar) { throw 'Windows tar.exe is required to unpack the OCR setup files.' }
            Invoke-Checked -Executable $tar.Source -ArgumentList @('-xf', $archive, '-C', $extractDirectory) -Name 'Micromamba unpacking'
            $micromambaFromArchive = Get-ChildItem -LiteralPath $extractDirectory -Filter 'micromamba.exe' -File -Recurse |
                Select-Object -First 1
            if (-not $micromambaFromArchive) { throw 'The micromamba download did not contain micromamba.exe.' }
            Copy-Item -LiteralPath $micromambaFromArchive.FullName -Destination $micromambaExecutable -Force
        }

        $env:MAMBA_ROOT_PREFIX = Join-Path $projectRoot '.mamba-ocr'
        $createArguments = @(
            'create', '--yes', '--prefix', $tesseractPrefix, '--channel', 'conda-forge', "tesseract=$tesseractVersion"
        )
        Invoke-Checked -Executable $micromambaExecutable -ArgumentList $createArguments -Name 'Tesseract OCR setup'

        $dataDirectory = Join-Path $tesseractPrefix 'share\tessdata'
        $requiredModels = @('chi_sim.traineddata', 'eng.traineddata', 'osd.traineddata')
        foreach ($model in $requiredModels) {
            if (-not (Test-Path -LiteralPath (Join-Path $dataDirectory $model) -PathType Leaf)) {
                throw "The OCR download is missing required language data: $model"
            }
        }

        Get-ChildItem -LiteralPath $dataDirectory -Filter '*.traineddata' -File |
            Where-Object { $_.Name -notin $requiredModels } |
            ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }

        $workingTesseract = Join-Path $tesseractPrefix 'Library\bin\tesseract.exe'
        if (-not (Test-Tesseract $workingTesseract $dataDirectory)) {
            throw 'Tesseract was installed but could not load its Chinese language data.'
        }
        Invoke-Checked -Executable $micromambaExecutable -ArgumentList @('clean', '--all', '--yes') -Name 'OCR cache cleanup'
        if (-not (Test-Tesseract $workingTesseract $dataDirectory)) {
            throw 'Tesseract could not be verified after removing the installer cache.'
        }
    }

    if (-not $workingTesseract) { throw 'No working Tesseract OCR executable was found.' }
    Write-Host 'Chinese OCR is ready.' -ForegroundColor Green

    $resolvedTemp = [System.IO.Path]::GetFullPath($tempRoot)
    $tempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
    $resolvedTempParent = [System.IO.Path]::GetDirectoryName($resolvedTemp)
    if ([string]::Equals($resolvedTempParent, $tempBase, [System.StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $resolvedTemp).StartsWith('ppt-takeaway-setup-', [System.StringComparison]::Ordinal)) {
        Remove-Item -LiteralPath $resolvedTemp -Recurse -Force
    }

    Write-Host "`nSetup complete. Run run_ppt_extractor.bat to start the app." -ForegroundColor Green
    exit 0
} catch {
    Write-Host "`nSetup failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'No system PATH or registry settings were changed.'
    Write-Host "Temporary downloads, if any, are in: $tempRoot"
    exit 1
}
