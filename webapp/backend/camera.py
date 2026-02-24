"""Camera abstraction layer with multi-backend support for Motic microscopes.

Tier 1: OpenCV (DirectShow/UVC) - works if Motic installs a DirectShow driver
Tier 2: Micro-Manager (pymmcore) - community Motic device adapter
Tier 3: Motic DLL wrapper (ctypes) - proprietary SDK, requires DLL from user
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, List
import numpy as np
import threading
import time


class CameraBackend(ABC):
    """Abstract camera backend interface."""

    name: str = "unknown"

    @abstractmethod
    def open(self, device_id: int = 0) -> bool:
        ...

    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        ...

    @abstractmethod
    def release(self) -> None:
        ...

    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        ...

    def set_property(self, prop: str, value: float) -> bool:
        return False

    def get_properties(self) -> Dict:
        return {}


class OpenCVBackend(CameraBackend):
    """Tier 1: OpenCV VideoCapture. Tries DirectShow first on Windows."""

    name = "opencv"

    def __init__(self):
        self._cap = None

    def open(self, device_id: int = 0) -> bool:
        import cv2

        # Try DirectShow backend first (Windows, best for Motic drivers)
        self._cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(device_id)
        if not self._cap.isOpened():
            return False
        # Set reasonable defaults
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        return True

    def read_frame(self) -> Optional[np.ndarray]:
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            return frame if ret else None
        return None

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def get_resolution(self) -> Tuple[int, int]:
        import cv2

        if not self._cap:
            return (0, 0)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def set_property(self, prop: str, value: float) -> bool:
        import cv2

        if not self._cap:
            return False
        prop_map = {
            "exposure": cv2.CAP_PROP_EXPOSURE,
            "gain": cv2.CAP_PROP_GAIN,
            "brightness": cv2.CAP_PROP_BRIGHTNESS,
            "contrast": cv2.CAP_PROP_CONTRAST,
            "white_balance": cv2.CAP_PROP_WB_TEMPERATURE,
        }
        cv_prop = prop_map.get(prop)
        if cv_prop is not None:
            return self._cap.set(cv_prop, value)
        return False


class MicroManagerBackend(CameraBackend):
    """Tier 2: Micro-Manager via pymmcore. Supports many microscope cameras."""

    name = "micromanager"

    def __init__(self):
        self._mmc = None

    def open(self, device_id: int = 0) -> bool:
        try:
            import pymmcore

            self._mmc = pymmcore.CMMCore()
            # User would need to configure MMConfig for their Motic camera
            # Try common Motic adapter names
            adapters = ["MoticCamera", "MoticUCam", "OpenCVgrabber"]
            for adapter in adapters:
                try:
                    self._mmc.loadDevice("Camera", "MoticCamera", adapter)
                    self._mmc.initializeDevice("Camera")
                    self._mmc.setCameraDevice("Camera")
                    return True
                except Exception:
                    continue
            return False
        except ImportError:
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        if not self._mmc:
            return None
        try:
            self._mmc.snapImage()
            img = self._mmc.getImage()
            if img.ndim == 2:
                import cv2

                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
        except Exception:
            return None

    def release(self) -> None:
        if self._mmc:
            try:
                self._mmc.reset()
            except Exception:
                pass
            self._mmc = None

    def get_resolution(self) -> Tuple[int, int]:
        if not self._mmc:
            return (0, 0)
        try:
            return (self._mmc.getImageWidth(), self._mmc.getImageHeight())
        except Exception:
            return (0, 0)


class MoticDLLBackend(CameraBackend):
    """Tier 3: Motic proprietary DLL via ctypes. Stub - requires actual DLL."""

    name = "motic_dll"

    def open(self, device_id: int = 0) -> bool:
        # This would wrap the Motic SDK DLL using ctypes
        # Requires: MotiConnect.dll or similar from Motic installation
        # Not implemented until DLL path and API are provided
        return False

    def read_frame(self) -> Optional[np.ndarray]:
        return None

    def release(self) -> None:
        pass

    def get_resolution(self) -> Tuple[int, int]:
        return (0, 0)


class CameraManager:
    """Orchestrates camera lifecycle with automatic backend selection."""

    BACKENDS = [OpenCVBackend, MicroManagerBackend, MoticDLLBackend]

    def __init__(self):
        self._backend: Optional[CameraBackend] = None
        self._is_running = False
        self._lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def backend_name(self) -> str:
        return self._backend.name if self._backend else "none"

    def start(self, device_id: int = 0, preferred_backend: Optional[str] = None) -> Dict:
        """Start camera. Tries backends in priority order."""
        with self._lock:
            if self._is_running:
                return {"status": "already_running", "backend": self.backend_name}

            backends_to_try = self.BACKENDS
            if preferred_backend:
                backends_to_try = [
                    b for b in self.BACKENDS if b.name == preferred_backend
                ] or self.BACKENDS

            for BackendClass in backends_to_try:
                backend = BackendClass()
                try:
                    if backend.open(device_id):
                        self._backend = backend
                        self._is_running = True
                        return {
                            "status": "ok",
                            "backend": backend.name,
                            "resolution": backend.get_resolution(),
                        }
                except Exception:
                    continue

            return {"status": "error", "message": "No camera backend could connect"}

    def stop(self):
        """Stop and release camera."""
        with self._lock:
            if self._backend:
                self._backend.release()
            self._backend = None
            self._is_running = False
            self._last_frame = None

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest camera frame (BGR numpy array)."""
        if not self._is_running or not self._backend:
            return None
        frame = self._backend.read_frame()
        if frame is not None:
            self._last_frame = frame
        return frame

    def capture_still(self) -> Optional[np.ndarray]:
        """Capture a full-resolution frame for analysis."""
        return self.get_frame()

    def get_status(self) -> Dict:
        return {
            "available": self._is_running,
            "backend": self.backend_name,
            "resolution": self._backend.get_resolution() if self._backend else None,
        }

    def set_property(self, prop: str, value: float) -> bool:
        if self._backend:
            return self._backend.set_property(prop, value)
        return False

    def release(self):
        self.stop()

    @staticmethod
    def list_available_backends() -> List[str]:
        """Check which backends are importable."""
        available = ["opencv"]  # cv2 is always available in this project
        try:
            import pymmcore  # noqa: F401
            available.append("micromanager")
        except ImportError:
            pass
        return available
