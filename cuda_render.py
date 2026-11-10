"""CUDA driver and NVRTC based rendering for face-detection overlays."""
import ctypes
from importlib import import_module
import threading
from typing import Any

import numpy as np

try:
    _driver: Any = import_module("cuda.bindings.driver")
    _nvrtc: Any = import_module("cuda.bindings.nvrtc")
except ImportError as exc:
    _driver = None
    _nvrtc = None
    _CUDA_IMPORT_ERROR = str(exc)
else:
    _CUDA_IMPORT_ERROR = None


_FONT_ROWS = np.zeros((128, 5), dtype=np.uint8)
for _character, _rows in {
    " ": (0, 0, 0, 0, 0),
    ".": (0, 0, 0, 0, 2),
    "0": (7, 5, 5, 5, 7),
    "1": (2, 6, 2, 2, 7),
    "2": (7, 1, 7, 4, 7),
    "3": (7, 1, 7, 1, 7),
    "4": (5, 5, 7, 1, 1),
    "5": (7, 4, 7, 1, 7),
    "6": (7, 4, 7, 5, 7),
    "7": (7, 1, 1, 1, 1),
    "8": (7, 5, 7, 5, 7),
    "9": (7, 5, 7, 1, 7),
    ":": (0, 2, 0, 2, 0),
    "F": (7, 4, 6, 4, 4),
    "a": (0, 3, 5, 7, 5),
    "c": (0, 3, 4, 4, 3),
    "e": (0, 7, 6, 4, 3),
    "s": (0, 3, 4, 2, 6),
}.items():
    _FONT_ROWS[ord(_character)] = _rows


_KERNEL_SOURCE = r"""
__device__ bool glyph_pixel(
    const unsigned char* font, unsigned char character, int x, int y
) {
    return x < 3 && y < 5
        && (font[(int)character * 5 + y] & (1 << (2 - x))) != 0;
}

__device__ void draw_text_pixel(
    const unsigned char* text,
    int text_length,
    int text_x,
    int text_y,
    const unsigned char* font,
    unsigned char* blue,
    unsigned char* green,
    unsigned char* red
) {
    int column = text_x / 8;
    int glyph_x = (text_x % 8) / 2;
    int glyph_y = text_y / 2;
    if (column < text_length
        && glyph_pixel(font, text[column], glyph_x, glyph_y)) {
        *blue = 0;
        *green = 0;
        *red = 0;
    }
}

extern "C" __global__ void draw_overlays(
    unsigned char* frame,
    int width,
    int height,
    const int* boxes,
    const unsigned char* labels,
    const int* label_lengths,
    int label_stride,
    int face_total,
    const unsigned char* count_text,
    int count_length,
    const unsigned char* font,
    int draw_count
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixel = (y * width + x) * 3;
    unsigned char blue = frame[pixel];
    unsigned char green = frame[pixel + 1];
    unsigned char red = frame[pixel + 2];

    for (int index = 0; index < face_total; ++index) {
        int x1 = boxes[index * 4];
        int y1 = boxes[index * 4 + 1];
        int x2 = boxes[index * 4 + 2];
        int y2 = boxes[index * 4 + 3];
        bool inside = x1 <= x && x < x2 && y1 <= y && y < y2;
        bool border = inside && (
            x - x1 < 2 || x2 - 1 - x < 2
            || y - y1 < 2 || y2 - 1 - y < 2
        );
        bool horizontal_corner = (
            (x < x1 + 20 || x >= x2 - 20)
            && (y - y1 < 3 || y2 - 1 - y < 3)
        );
        bool vertical_corner = (
            (y < y1 + 20 || y >= y2 - 20)
            && (x - x1 < 3 || x2 - 1 - x < 3)
        );
        if (border || (inside && (horizontal_corner || vertical_corner))) {
            blue = 0;
            green = 255;
            red = 0;
        }

        int label_top = y1 - 14;
        if (label_top < 0) label_top = y1;
        int label_length = label_lengths[index];
        int label_width = label_length * 8 + 4;
        if (x1 <= x && x < x1 + label_width
            && label_top <= y && y < label_top + 14) {
            blue = 0;
            green = 255;
            red = 0;
            draw_text_pixel(
                labels + index * label_stride,
                label_length,
                x - x1 - 2,
                y - label_top - 2,
                font,
                &blue,
                &green,
                &red
            );
        }
    }

    if (draw_count && 10 <= x && x < 10 + count_length * 8 + 4
        && 8 <= y && y < 22) {
        blue = 0;
        green = 255;
        red = 0;
        draw_text_pixel(
            count_text,
            count_length,
            x - 12,
            y - 10,
            font,
            &blue,
            &green,
            &red
        );
    }

    frame[pixel] = blue;
    frame[pixel + 1] = green;
    frame[pixel + 2] = red;
}
"""

_renderer_state = threading.local()
_device_probe = None
_device_error = None


def _check_cuda(result, operation, success):
    status, *values = result
    if status != success:
        raise RuntimeError(f"{operation} failed: {status}")
    return values


def cuda_overlay_available():
    """Return whether a CUDA device is available to the CUDA driver."""
    global _device_probe, _device_error
    if _device_probe is not None:
        return _device_probe
    if _driver is None:
        _device_error = _CUDA_IMPORT_ERROR
        _device_probe = False
        return False

    status, = _driver.cuInit(0)
    if status != _driver.CUresult.CUDA_SUCCESS:
        _device_error = f"CUDA driver initialization failed: {status}"
        _device_probe = False
        return False
    status, count = _driver.cuDeviceGetCount()
    if status != _driver.CUresult.CUDA_SUCCESS:
        _device_error = f"CUDA device query failed: {status}"
        _device_probe = False
        return False
    _device_probe = count > 0
    if not _device_probe:
        _device_error = "no CUDA device is available"
    return _device_probe


def cuda_overlay_status():
    """Describe the active overlay backend for startup logging."""
    if cuda_overlay_available():
        return "CUDA overlay rendering"
    return f"CPU overlay rendering ({_device_error or 'CUDA unavailable'})"


class _CudaRenderer:
    def __init__(self):
        self.driver = _driver
        self.nvrtc = _nvrtc
        success = self.driver.CUresult.CUDA_SUCCESS
        self.device_buffers = {}
        self.context = None
        self.module = None
        self.program = None

        initialized = False
        try:
            _check_cuda(
                self.driver.cuInit(0), "CUDA driver initialization", success
            )
            device = _check_cuda(
                self.driver.cuDeviceGet(0), "CUDA device selection", success
            )[0]
            self.context = _check_cuda(
                self.driver.cuCtxCreate(None, 0, device),
                "CUDA context creation",
                success,
            )[0]
            self.module = self._compile_kernel(device)
            self.kernel = _check_cuda(
                self.driver.cuModuleGetFunction(self.module, b"draw_overlays"),
                "CUDA kernel lookup",
                success,
            )[0]
            initialized = True
        finally:
            if not initialized:
                self.close()

    def _compile_kernel(self, device):
        success = self.nvrtc.nvrtcResult.NVRTC_SUCCESS
        major = _check_cuda(
            self.driver.cuDeviceGetAttribute(
                self.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device,
            ),
            "CUDA compute capability query",
            self.driver.CUresult.CUDA_SUCCESS,
        )[0]
        minor = _check_cuda(
            self.driver.cuDeviceGetAttribute(
                self.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                device,
            ),
            "CUDA compute capability query",
            self.driver.CUresult.CUDA_SUCCESS,
        )[0]
        self.program = _check_cuda(
            self.nvrtc.nvrtcCreateProgram(
                _KERNEL_SOURCE.encode("ascii"),
                b"face_overlay.cu",
                0,
                None,
                None,
            ),
            "CUDA kernel program creation",
            success,
        )[0]
        status, = self.nvrtc.nvrtcCompileProgram(
            self.program,
            1,
            [f"--gpu-architecture=compute_{major}{minor}".encode("ascii")],
        )
        if status != success:
            raise RuntimeError(f"CUDA kernel compilation failed: {status}")

        ptx_size = _check_cuda(
            self.nvrtc.nvrtcGetPTXSize(self.program),
            "CUDA kernel PTX sizing",
            success,
        )[0]
        ptx = bytes(ptx_size)
        _check_cuda(
            self.nvrtc.nvrtcGetPTX(self.program, ptx),
            "CUDA kernel PTX retrieval",
            success,
        )
        _check_cuda(
            self.nvrtc.nvrtcDestroyProgram(self.program),
            "CUDA compiler cleanup",
            success,
        )
        self.program = None
        return _check_cuda(
            self.driver.cuModuleLoadData(ptx),
            "CUDA kernel module loading",
            self.driver.CUresult.CUDA_SUCCESS,
        )[0]

    def _upload(self, name, data):
        data = np.ascontiguousarray(data)
        size = max(data.nbytes, 1)
        entry = self.device_buffers.get(name)
        if entry is None or entry[1] < size:
            if entry is not None:
                _check_cuda(
                    self.driver.cuMemFree(entry[0]),
                    f"CUDA {name} buffer release",
                    self.driver.CUresult.CUDA_SUCCESS,
                )
            device_ptr = _check_cuda(
                self.driver.cuMemAlloc(size),
                f"CUDA {name} buffer allocation",
                self.driver.CUresult.CUDA_SUCCESS,
            )[0]
            entry = (device_ptr, size)
            self.device_buffers[name] = entry
        if data.nbytes:
            _check_cuda(
                self.driver.cuMemcpyHtoD(entry[0], data, data.nbytes),
                f"CUDA {name} upload",
                self.driver.CUresult.CUDA_SUCCESS,
            )
        return entry[0]

    def render(self, frame, faces, face_count):
        height, width = (int(value) for value in frame.shape[:2])
        host_frame = np.ascontiguousarray(frame)
        face_total = len(faces)
        boxes = np.zeros((max(face_total, 1), 4), dtype=np.int32)
        if face_total:
            boxes[:face_total] = np.asarray(
                [face[:4] for face in faces], dtype=np.int32
            )

        labels = [f"Face: {float(face[4]):.2f}" for face in faces]
        label_lengths = np.zeros((max(face_total, 1),), dtype=np.int32)
        if face_total:
            label_lengths[:face_total] = [len(label) for label in labels]
        label_stride = int(max(max(label_lengths, default=0), 1))
        label_chars = np.zeros((max(face_total, 1), label_stride), dtype=np.uint8)
        for index, label in enumerate(labels):
            label_chars[index, :len(label)] = np.frombuffer(
                label.encode("ascii"), dtype=np.uint8
            )

        count_text = (
            f"Faces: {face_count}" if face_count is not None else ""
        ).encode("ascii")
        count_chars = np.frombuffer(count_text or b"\0", dtype=np.uint8)
        frame_ptr = self._upload("frame", host_frame)
        boxes_ptr = self._upload("boxes", boxes)
        labels_ptr = self._upload("labels", label_chars)
        lengths_ptr = self._upload("label_lengths", label_lengths)
        count_ptr = self._upload("count", count_chars)
        font_ptr = self._upload("font", _FONT_ROWS)

        arguments = [
            ctypes.c_uint64(int(frame_ptr)),
            ctypes.c_int(width),
            ctypes.c_int(height),
            ctypes.c_uint64(int(boxes_ptr)),
            ctypes.c_uint64(int(labels_ptr)),
            ctypes.c_uint64(int(lengths_ptr)),
            ctypes.c_int(label_stride),
            ctypes.c_int(face_total),
            ctypes.c_uint64(int(count_ptr)),
            ctypes.c_int(len(count_text)),
            ctypes.c_uint64(int(font_ptr)),
            ctypes.c_int(face_count is not None),
        ]
        kernel_params = (ctypes.c_void_p * len(arguments))(
            *(ctypes.addressof(argument) for argument in arguments)
        )
        block_width = 16
        block_height = 16
        _check_cuda(
            self.driver.cuLaunchKernel(
                self.kernel,
                (width + block_width - 1) // block_width,
                (height + block_height - 1) // block_height,
                1,
                block_width,
                block_height,
                1,
                0,
                0,
                kernel_params,
                0,
            ),
            "CUDA overlay kernel launch",
            self.driver.CUresult.CUDA_SUCCESS,
        )
        _check_cuda(
            self.driver.cuMemcpyDtoH(host_frame, frame_ptr, host_frame.nbytes),
            "CUDA rendered frame download",
            self.driver.CUresult.CUDA_SUCCESS,
        )
        frame[...] = host_frame
        return frame

    def close(self):
        if self.context is None:
            return
        for name, (device_ptr, _) in self.device_buffers.items():
            _check_cuda(
                self.driver.cuMemFree(device_ptr),
                f"CUDA {name} buffer release",
                self.driver.CUresult.CUDA_SUCCESS,
            )
        self.device_buffers.clear()
        if self.module is not None:
            _check_cuda(
                self.driver.cuModuleUnload(self.module),
                "CUDA kernel module cleanup",
                self.driver.CUresult.CUDA_SUCCESS,
            )
            self.module = None
        if self.program is not None:
            _check_cuda(
                self.nvrtc.nvrtcDestroyProgram(self.program),
                "CUDA compiler cleanup",
                self.nvrtc.nvrtcResult.NVRTC_SUCCESS,
            )
            self.program = None
        _check_cuda(
            self.driver.cuCtxDestroy(self.context),
            "CUDA context cleanup",
            self.driver.CUresult.CUDA_SUCCESS,
        )
        self.context = None


def render_overlays_cuda(frame, faces, face_count=None):
    """Draw face boxes, labels, and an optional face count using CUDA."""
    renderer = getattr(_renderer_state, "renderer", None)
    if renderer is None:
        renderer = _CudaRenderer()
        _renderer_state.renderer = renderer
    return renderer.render(frame, faces, face_count)


def close_cuda_renderer():
    """Release the CUDA context owned by the current thread, if any."""
    renderer = getattr(_renderer_state, "renderer", None)
    if renderer is not None:
        renderer.close()
        del _renderer_state.renderer
