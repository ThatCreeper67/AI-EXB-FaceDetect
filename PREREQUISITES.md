# Prerequisites for Building OpenCV with CUDA on Windows

## Required Software (Install in Order)

### 1. Visual Studio 2022 Community (Free)
- Download: https://visualstudio.microsoft.com/downloads/
- **Workload**: "Desktop development with C++" (mandatory)
- Individual components:
  - MSVC v143 - VS 2022 C++ x64/x86 build tools
  - Windows 11 SDK (or latest)
  - C++ CMake tools for Windows

### 2. CMake 3.25+
- Download: https://cmake.org/download/
- **Important**: Check "Add CMake to system PATH for all users" during install
- Verify: `cmake --version` in new terminal

### 3. CUDA Toolkit 12.8+ (Required for RTX 5050 Blackwell)
- Download: https://developer.nvidia.com/cuda-12-8-0-download-archive
- Select: Windows → x86_64 → 11/12 → exe (local)
- **Important**: After install, verify:
  - `nvcc --version` shows 12.8+
  - `echo %CUDA_PATH%` points to install dir (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`)

### 4. cuDNN 9.x for CUDA 12
- Download: https://developer.nvidia.com/cudnn (requires NVIDIA account)
- Select: cuDNN v9.x.x for CUDA 12.x
- Extract and copy files to CUDA directory:
  - `bin\cudnn*.dll` → `%CUDA_PATH%\bin\`
  - `include\cudnn*.h` → `%CUDA_PATH%\include\`
  - `lib\x64\cudnn*.lib` → `%CUDA_PATH%\lib\x64\`

### 5. Git for Windows
- Download: https://git-scm.com/download/win
- Verify: `git --version`

### 6. Python 3.15 (Already installed)
- Ensure numpy is installed: `pip install numpy`

---

## After Installing Prerequisites

1. **Open a NEW PowerShell terminal** (as Administrator recommended)
2. Verify all tools are in PATH:
   ```powershell
   cmake --version
   nvcc --version
   git --version
   python --version
   ```

3. Run the build script:
   ```powershell
   cd "C:\Nerd\AI EXB"
   .\build_opencv_cuda.ps1
   ```

---

## Expected Build Time & Resources

| Phase | Time | Disk Space |
|-------|------|------------|
| Download sources | 2-5 min | ~1 GB |
| CMake configure | 1-2 min | ~500 MB |
| Build (Release) | 30-60 min | ~8-10 GB |
| Install | 1-2 min | ~2 GB |
| **Total** | **35-70 min** | **~12 GB** |

---

## RTX 5050 (Blackwell) Specific Notes

- **CUDA 12.8+ required** for Blackwell architecture support
- **Compute Capability 12.0** (sm_120) for RTX 5000 series
- The CMake script includes: `-DCUDA_ARCH_BIN=86;89;90;120` and `-DCUDA_ARCH_PTX=120`
- This covers: Ampere (RTX 30), Ada (RTX 40), Blackwell (RTX 50)

---

## Troubleshooting

### "CUDA not found"
- Restart terminal after CUDA install
- Check `nvcc --version` works
- Ensure `CUDA_PATH` environment variable is set

### "Visual Studio not found"
- Install "Desktop development with C++" workload
- Run Visual Studio Installer → Modify → check the workload

### Build fails with memory errors
- Use `-j` flag to limit parallel jobs: `msbuild OpenCV.sln /p:Configuration=Release /m:4`
- Close other applications

### Python module not found after install
- Find the built cv2 directory: `dir C:\opencv_install\**\cv2 /s`
- Add to PYTHONPATH or copy to site-packages

---

## Alternative: Pre-built OpenCV with CUDA

If building fails, consider these alternatives:
1. **OpenCV AI Kit (OAK)** - pre-built wheels with CUDA
2. **Use PyTorch with ONNX export** - convert your Caffe model to ONNX, run with PyTorch CUDA
3. **Docker with OpenCV CUDA** - `docker pull opencv/opencv:4.10-cuda`

---

## CUDA Overlay Rendering

Face boxes, confidence labels, and the face count can be drawn by a custom CUDA
kernel independently of OpenCV's CUDA build or a C++ compiler. Install the
optional Python CUDA 13.5 binding with:

```powershell
python -m pip install -r requirements-cuda.txt
```

The application selects CUDA overlay rendering when the CUDA driver can access
a device. Otherwise, it keeps using the existing OpenCV CPU drawing path. This
does not enable CUDA acceleration for face detection; that still requires an
OpenCV build with CUDA and DNN CUDA support.

---

## Quick Test After Build

```powershell
python -c "
import cv2
print(f'OpenCV: {cv2.__version__}')
print(f'CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}')
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    dev = cv2.cuda.DeviceInfo()
    print(f'Device: {dev.name()}')
    print(f'Compute: {dev.major()}.{dev.minor()}')
"
```

Expected output with RTX 5050:
```
OpenCV: 4.10.0
CUDA devices: 1
Device: NVIDIA GeForce RTX 5050
Compute: 12.0
```