<#
.SYNOPSIS
    Build OpenCV with CUDA support for Python 3.15 on Windows
.DESCRIPTION
    Compiles OpenCV 4.x with CUDA 12.8+, cuDNN, and Python bindings.
    Required for RTX 5050 (Blackwell) support.
.NOTES
    Prerequisites:
    - Visual Studio 2022 with "Desktop development with C++" workload
    - CUDA Toolkit 12.8+ (for Blackwell/RTX 5000 series)
    - cuDNN 9.x for CUDA 12
    - CMake 3.25+
    - Python 3.15 with development headers
    - Git for Windows
    - ~10-15 GB free disk space
    - Build time: 30-60 minutes
#>

param(
    [string]$OpenCVVersion = "4.10.0",
    [string]$OpenCVContribVersion = "4.10.0",
    [string]$BuildDir = "C:\opencv_build",
    [string]$InstallDir = "C:\opencv_install",
    [string]$PythonExe = (Get-Command python).Source,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

Write-Host "=== OpenCV with CUDA Build Script ===" -ForegroundColor Green
Write-Host "OpenCV Version: $OpenCVVersion"
Write-Host "Build Dir: $BuildDir"
Write-Host "Install Dir: $InstallDir"
Write-Host "Python: $PythonExe"

# Check prerequisites
function Check-Prerequisites {
    Write-Host "`n=== Checking Prerequisites ===" -ForegroundColor Yellow

    # Visual Studio
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vs = & $vswhere -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vs) {
            Write-Host "✓ Visual Studio: $vs" -ForegroundColor Green
        } else {
            Write-Host "✗ Visual Studio with C++ workload not found" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "✗ vswhere not found - install Visual Studio 2022" -ForegroundColor Red
        exit 1
    }

    # CUDA
    if ($env:CUDA_PATH -and (Test-Path $env:CUDA_PATH)) {
        $cudaVersion = (Get-Item "$env:CUDA_PATH\version.txt").VersionInfo.FileVersion
        Write-Host "✓ CUDA Toolkit: $cudaVersion at $env:CUDA_PATH" -ForegroundColor Green
        if ([version]$cudaVersion -lt [version]"12.8") {
            Write-Host "⚠ Warning: CUDA 12.8+ recommended for RTX 5050 (Blackwell)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "✗ CUDA Toolkit not found in CUDA_PATH" -ForegroundColor Red
        exit 1
    }

    # cuDNN
    $cudnnPath = "$env:CUDA_PATH\include\cudnn_version.h"
    if (Test-Path $cudnnPath) {
        $content = Get-Content $cudnnPath
        $major = ($content | Select-String "CUDNN_MAJOR\s+(\d+)").Matches.Groups[1].Value
        $minor = ($content | Select-String "CUDNN_MINOR\s+(\d+)").Matches.Groups[1].Value
        $patch = ($content | Select-String "CUDNN_PATCHLEVEL\s+(\d+)").Matches.Groups[1].Value
        Write-Host "✓ cuDNN: $major.$minor.$patch" -ForegroundColor Green
    } else {
        Write-Host "⚠ cuDNN not found in CUDA_PATH - download from NVIDIA" -ForegroundColor Yellow
    }

    # CMake
    $cmake = Get-Command cmake -ErrorAction SilentlyContinue
    if ($cmake) {
        Write-Host "✓ CMake: $($cmake.Version)" -ForegroundColor Green
    } else {
        Write-Host "✗ CMake not found - install from cmake.org" -ForegroundColor Red
        exit 1
    }

    # Python
    $pyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    Write-Host "✓ Python: $pyVersion"

    $pyInclude = python -c "import sysconfig; print(sysconfig.get_path('include'))"
    $pyLib = python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
    Write-Host "  Include: $pyInclude"
    Write-Host "  Library: $pyLib"

    # Git
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($git) {
        Write-Host "✓ Git: $($git.Version)" -ForegroundColor Green
    } else {
        Write-Host "✗ Git not found" -ForegroundColor Red
        exit 1
    }
}

function Download-Sources {
    Write-Host "`n=== Downloading OpenCV Sources ===" -ForegroundColor Yellow

    if (-not (Test-Path "$BuildDir\opencv")) {
        Write-Host "Cloning OpenCV..."
        git clone --depth 1 --branch $OpenCVVersion https://github.com/opencv/opencv.git "$BuildDir\opencv"
    } else {
        Write-Host "OpenCV already exists, updating..."
        cd "$BuildDir\opencv"
        git fetch --depth 1 origin $OpenCVVersion
        git checkout $OpenCVVersion
    }

    if (-not (Test-Path "$BuildDir\opencv_contrib")) {
        Write-Host "Cloning OpenCV Contrib..."
        git clone --depth 1 --branch $OpenCVContribVersion https://github.com/opencv/opencv_contrib.git "$BuildDir\opencv_contrib"
    } else {
        Write-Host "OpenCV Contrib already exists, updating..."
        cd "$BuildDir\opencv_contrib"
        git fetch --depth 1 origin $OpenCVContribVersion
        git checkout $OpenCVContribVersion
    }
}

function Configure-CMake {
    Write-Host "`n=== Configuring CMake ===" -ForegroundColor Yellow

    $buildType = "Release"
    $cmakeBuildDir = "$BuildDir\build"

    if ($Clean -and (Test-Path $cmakeBuildDir)) {
        Write-Host "Cleaning build directory..."
        Remove-Item -Recurse -Force $cmakeBuildDir
    }

    New-Item -ItemType Directory -Force -Path $cmakeBuildDir | Out-Null
    cd $cmakeBuildDir

    # Find Python library
    $pyLibName = "python$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")"
    $pyLibPath = "$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")\$pyLibName.lib"

    $cmakeArgs = @(
        "-G", "Visual Studio 17 2022",
        "-A", "x64",
        "-DCMAKE_BUILD_TYPE=$buildType",
        "-DCMAKE_INSTALL_PREFIX=$InstallDir",
        "-DOPENCV_EXTRA_MODULES_PATH=$BuildDir\opencv_contrib\modules",

        # CUDA
        "-DWITH_CUDA=ON",
        "-DCUDA_FAST_MATH=ON",
        "-DWITH_CUBLAS=ON",
        "-DCUDA_ARCH_BIN=86;89;90;120",  # Ampere, Ada, Blackwell
        "-DCUDA_ARCH_PTX=120",             # PTX for future compat (Blackwell)

        # cuDNN
        "-DWITH_CUDNN=ON",

        # Python
        "-DBUILD_opencv_python3=ON",
        "-DBUILD_opencv_python2=OFF",
        "-DPYTHON3_EXECUTABLE=$PythonExe",
        "-DPYTHON3_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")",
        "-DPYTHON3_LIBRARY=$pyLibPath",
        "-DPYTHON3_NUMPY_INCLUDE_DIRS=$(python -c "import numpy; print(numpy.get_include())")",
        "-DPYTHON3_PACKAGES_PATH=$(python -c "import site; print(site.getsitepackages()[0])")",

        # Performance
        "-DENABLE_FAST_MATH=ON",
        "-DWITH_OPENMP=ON",
        "-DWITH_TBB=ON",

        # Features
        "-DWITH_DNN=ON",
        "-DOPENCV_DNN_CUDA=ON",
        "-DWITH_OPENCL=OFF",  # Prefer CUDA

        # Disable heavy modules
        "-DBUILD_DOCS=OFF",
        "-DBUILD_EXAMPLES=OFF",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_PERF_TESTS=OFF",
        "-DBUILD_opencv_apps=OFF",

        # Video I/O
        "-DWITH_FFMPEG=ON",
        "-DWITH_MSMF=ON",

        "$BuildDir\opencv"
    )

    Write-Host "Running CMake..." -ForegroundColor Cyan
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configure failed!" -ForegroundColor Red
        exit 1
    }
}

function Build-OpenCV {
    Write-Host "`n=== Building OpenCV ===" -ForegroundColor Yellow

    cd "$BuildDir\build"

    Write-Host "Building with MSBuild..." -ForegroundColor Cyan
    & msbuild "OpenCV.sln" /p:Configuration=Release /m
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed!" -ForegroundColor Red
        exit 1
    }

    Write-Host "Installing..." -ForegroundColor Cyan
    & msbuild "INSTALL.vcxproj" /p:Configuration=Release
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Install failed!" -ForegroundColor Red
        exit 1
    }
}

function Test-Installation {
    Write-Host "`n=== Testing Installation ===" -ForegroundColor Yellow

    # Add install path to Python path
    $pySitePackages = "$InstallDir\python\cv2"
    if (Test-Path $pySitePackages) {
        $env:PYTHONPATH = "$pySitePackages;$env:PYTHONPATH"
    } else {
        # Find the cv2 directory
        $cv2Dirs = Get-ChildItem -Recurse -Directory -Path $InstallDir -Filter "cv2" -ErrorAction SilentlyContinue
        if ($cv2Dirs) {
            $env:PYTHONPATH = "$($cv2Dirs[0].FullName);$env:PYTHONPATH"
        }
    }

    # Also add DLL path
    $binDir = "$InstallDir\x64\Release"
    if (Test-Path $binDir) {
        $env:PATH = "$binDir;$env:PATH"
    }

    Write-Host "Testing Python import..." -ForegroundColor Cyan
    python -c "
import cv2
print(f'OpenCV version: {cv2.__version__}')
print(f'CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}')
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print(f'CUDA Device: {cv2.cuda.getDevice()}')
    print(f'CUDA Compute Capability: {cv2.cuda.DeviceInfo().major()}.{cv2.cuda.DeviceInfo().minor()}')
print('Build info (CUDA):')
for line in cv2.getBuildInformation().split('\n'):
    if 'CUDA' in line.upper() and ('ON' in line or 'YES' in line or '12.' in line):
        print(f'  {line.strip()}')
"

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ OpenCV with CUDA built and tested successfully!" -ForegroundColor Green
        Write-Host "To use in your project, add to Python path:" -ForegroundColor Cyan
        Write-Host "  $env:PYTHONPATH = \"$env:PYTHONPATH;$pySitePackages\""
        Write-Host "  \$env:PATH += \";$binDir\""
    } else {
        Write-Host "`n✗ Test failed" -ForegroundColor Red
        exit 1
    }
}

# Main
try {
    Check-Prerequisites
    Download-Sources
    Configure-CMake
    Build-OpenCV
    Test-Installation
} catch {
    Write-Host "`nBuild failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Build Complete ===" -ForegroundColor Green