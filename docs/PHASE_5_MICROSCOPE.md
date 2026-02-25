# Phase 5: Microscope Camera Integration

**Timeline**: Weeks 11-12 (14 days)  
**Priority**: Medium - Advanced feature for real-time analysis  
**Estimated Effort**: 30-40 hours  
**Status**: Foundation phase (full integration depends on hardware availability)

## Objectives
1. Research common microscope camera interfaces and SDKs
2. Design camera abstraction layer for multiple camera types
3. Implement camera capture and live preview
4. Add real-time analysis mode
5. Implement auto-focus and exposure optimization
6. Create time-lapse and batch capture features
7. Document hardware setup and integration

## Deliverables
- [ ] Camera interface abstraction layer
- [ ] Support for 2-3 common camera types
- [ ] Live preview with real-time analysis
- [ ] Camera settings optimization
- [ ] Batch capture workflow
- [ ] Hardware setup documentation
- [ ] Example integration scripts

---

## Common Microscope Cameras

### USB Webcams & Scientific Cameras
- **USB Video Class (UVC)**: Standard USB cameras
  - Libraries: OpenCV, PyUSB
  - Pros: Universal, no special drivers
  - Cons: Limited control, variable quality

- **GigE Vision**: Industrial cameras (Basler, FLIR, AVT)
  - Libraries: Vimba (Basler), Spinnaker (FLIR)
  - Pros: High performance, precise control
  - Cons: Requires specific SDKs

- **USB3 Vision**: USB3 industrial cameras
  - Libraries: Vendor-specific SDKs
  - Pros: Good throughput, easier setup than GigE
  - Cons: Platform-dependent

### Microscope-Specific
- **AmScope Cameras**: Common in research labs
  - Software: ToupView SDK, AmScope SDK
  - Interface: USB 2.0/3.0

- **Nikon/Olympus/Zeiss**: High-end microscopes
  - Proprietary SDKs and drivers
  - Often expensive but very capable

---

## Task Breakdown

### Task 5.1: Camera Abstraction Layer (Day 1-3)
**Goal**: Create unified interface for different camera types

**Design Pattern**: Abstract base class with specific implementations

**Implementation**: `microscope/camera_interface.py`
```python
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict
import numpy as np

class CameraInterface(ABC):
    """
    Abstract base class for microscope cameras
    """
    
    def __init__(self, camera_id: Optional[str] = None):
        self.camera_id = camera_id
        self.is_open = False
        self.current_settings = {}
    
    @abstractmethod
    def open(self) -> bool:
        """
        Open camera connection
        Returns: True if successful
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close camera connection"""
        pass
    
    @abstractmethod
    def capture_image(self) -> np.ndarray:
        """
        Capture single image
        Returns: NumPy array (H, W, 3) in RGB format
        """
        pass
    
    @abstractmethod
    def get_live_frame(self) -> np.ndarray:
        """
        Get current frame for live preview (may be lower res)
        Returns: NumPy array
        """
        pass
    
    @abstractmethod
    def set_exposure(self, exposure_ms: float):
        """Set exposure time in milliseconds"""
        pass
    
    @abstractmethod
    def set_gain(self, gain: float):
        """Set gain (0-100)"""
        pass
    
    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """Get image resolution (width, height)"""
        pass
    
    @abstractmethod
    def set_resolution(self, width: int, height: int):
        """Set image resolution"""
        pass
    
    def get_camera_info(self) -> Dict:
        """Get camera information"""
        return {
            'camera_id': self.camera_id,
            'resolution': self.get_resolution(),
            'settings': self.current_settings
        }
    
    def auto_expose(self, target_brightness: int = 128) -> float:
        """
        Automatically adjust exposure to target brightness
        
        Args:
            target_brightness: Target mean pixel value (0-255)
        
        Returns:
            Optimal exposure time
        """
        # Binary search for optimal exposure
        exposure_min = 1.0
        exposure_max = 100.0
        
        for _ in range(10):
            # Try middle exposure
            exposure = (exposure_min + exposure_max) / 2
            self.set_exposure(exposure)
            
            # Capture test image
            img = self.capture_image()
            brightness = np.mean(img)
            
            if abs(brightness - target_brightness) < 10:
                break
            elif brightness < target_brightness:
                exposure_min = exposure
            else:
                exposure_max = exposure
        
        return exposure
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
```

**Implementation 1: OpenCV (USB/Webcam)**: `microscope/opencv_camera.py`
```python
import cv2
import numpy as np
from camera_interface import CameraInterface

class OpenCVCamera(CameraInterface):
    """
    Implementation for USB/webcam using OpenCV
    Works with most USB Video Class (UVC) cameras
    """
    
    def __init__(self, camera_id: int = 0):
        super().__init__(camera_id)
        self.cap = None
    
    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            return False
        
        self.is_open = True
        
        # Set initial settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        return True
    
    def close(self):
        if self.cap:
            self.cap.release()
        self.is_open = False
    
    def capture_image(self) -> np.ndarray:
        if not self.is_open:
            raise RuntimeError("Camera not opened")
        
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image")
        
        # Convert BGR to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def get_live_frame(self) -> np.ndarray:
        return self.capture_image()
    
    def set_exposure(self, exposure_ms: float):
        # OpenCV exposure is typically in log scale
        # This is camera-dependent and may not work on all cameras
        if self.cap:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_ms)
            self.current_settings['exposure'] = exposure_ms
    
    def set_gain(self, gain: float):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_GAIN, gain)
            self.current_settings['gain'] = gain
    
    def get_resolution(self) -> Tuple[int, int]:
        if not self.cap:
            return (0, 0)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def set_resolution(self, width: int, height: int):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Usage
camera = OpenCVCamera(camera_id=0)
camera.open()
image = camera.capture_image()
camera.close()
```

**Implementation 2: GigE/USB3 Industrial Camera** (template)
```python
# Example for Basler cameras using pypylon
from pypylon import pylon
from camera_interface import CameraInterface

class BaslerCamera(CameraInterface):
    """
    Implementation for Basler GigE/USB3 cameras
    Requires: pip install pypylon
    """
    
    def __init__(self, camera_id: Optional[str] = None):
        super().__init__(camera_id)
        self.camera = None
        
    def open(self) -> bool:
        try:
            # Get first available camera if no ID specified
            if self.camera_id is None:
                self.camera = pylon.InstantCamera(
                    pylon.TlFactory.GetInstance().CreateFirstDevice()
                )
            else:
                # Find camera by serial number
                devices = pylon.TlFactory.GetInstance().EnumerateDevices()
                for device in devices:
                    if device.GetSerialNumber() == self.camera_id:
                        self.camera = pylon.InstantCamera(
                            pylon.TlFactory.GetInstance().CreateDevice(device)
                        )
                        break
            
            if not self.camera:
                return False
            
            self.camera.Open()
            self.is_open = True
            
            # Configure camera
            self.camera.PixelFormat = "RGB8"
            
            return True
            
        except Exception as e:
            print(f"Failed to open camera: {e}")
            return False
    
    def close(self):
        if self.camera:
            self.camera.Close()
        self.is_open = False
    
    def capture_image(self) -> np.ndarray:
        if not self.is_open:
            raise RuntimeError("Camera not opened")
        
        self.camera.StartGrabbing(1)  # Grab 1 image
        grab_result = self.camera.RetrieveResult(5000)  # 5s timeout
        
        if grab_result.GrabSucceeded():
            image = grab_result.Array
            grab_result.Release()
            return image
        else:
            raise RuntimeError("Image capture failed")
    
    def set_exposure(self, exposure_ms: float):
        if self.camera:
            self.camera.ExposureTime.SetValue(exposure_ms * 1000)  # microseconds
            self.current_settings['exposure'] = exposure_ms
    
    def set_gain(self, gain: float):
        if self.camera:
            self.camera.Gain.SetValue(gain)
            self.current_settings['gain'] = gain
    
    # ... other methods
```

**Camera Factory**: `microscope/camera_factory.py`
```python
from typing import Optional
from camera_interface import CameraInterface
from opencv_camera import OpenCVCamera
# from basler_camera import BaslerCamera  # If available
# from flir_camera import FLIRCamera       # If available

def create_camera(camera_type: str, camera_id: Optional[str] = None) -> CameraInterface:
    """
    Factory function to create appropriate camera instance
    
    Args:
        camera_type: 'opencv', 'basler', 'flir', etc.
        camera_id: Camera identifier (device ID, serial number, etc.)
    
    Returns:
        CameraInterface instance
    """
    camera_type = camera_type.lower()
    
    if camera_type == 'opencv':
        return OpenCVCamera(int(camera_id) if camera_id else 0)
    elif camera_type == 'basler':
        try:
            from basler_camera import BaslerCamera
            return BaslerCamera(camera_id)
        except ImportError:
            raise RuntimeError("Basler camera support not installed. Install pypylon.")
    elif camera_type == 'flir':
        try:
            from flir_camera import FLIRCamera
            return FLIRCamera(camera_id)
        except ImportError:
            raise RuntimeError("FLIR camera support not installed. Install PySpin.")
    else:
        raise ValueError(f"Unknown camera type: {camera_type}")

# Usage
camera = create_camera('opencv', camera_id='0')
```

**Deliverable**: Flexible camera abstraction supporting multiple types

---

### Task 5.2: Live Preview Mode (Day 4-6)
**Goal**: Real-time camera feed with analysis overlay

**Implementation**: `microscope/live_preview.py`
```python
import cv2
import numpy as np
from threading import Thread, Lock
from queue import Queue
import time

class LivePreview:
    """
    Real-time camera preview with optional analysis
    """
    
    def __init__(self, camera: CameraInterface, pipeline=None):
        self.camera = camera
        self.pipeline = pipeline
        self.is_running = False
        self.frame_lock = Lock()
        self.current_frame = None
        self.analysis_results = None
        
        # Performance tracking
        self.fps = 0
        self.analysis_fps = 0
        
    def start(self, enable_analysis: bool = False, analysis_interval: float = 1.0):
        """
        Start live preview
        
        Args:
            enable_analysis: Run cell detection on frames
            analysis_interval: Seconds between analyses
        """
        self.is_running = True
        
        # Start capture thread
        capture_thread = Thread(target=self._capture_loop, daemon=True)
        capture_thread.start()
        
        # Start analysis thread if enabled
        if enable_analysis and self.pipeline:
            analysis_thread = Thread(
                target=self._analysis_loop,
                args=(analysis_interval,),
                daemon=True
            )
            analysis_thread.start()
        
        # Display loop (runs in main thread)
        self._display_loop()
    
    def stop(self):
        """Stop live preview"""
        self.is_running = False
    
    def _capture_loop(self):
        """Continuously capture frames"""
        last_time = time.time()
        frame_count = 0
        
        while self.is_running:
            try:
                frame = self.camera.get_live_frame()
                
                with self.frame_lock:
                    self.current_frame = frame
                
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    self.fps = frame_count / (current_time - last_time)
                    frame_count = 0
                    last_time = current_time
                
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.1)
    
    def _analysis_loop(self, interval: float):
        """Periodically analyze frames"""
        last_analysis = 0
        analysis_count = 0
        
        while self.is_running:
            current_time = time.time()
            
            if current_time - last_analysis >= interval:
                with self.frame_lock:
                    if self.current_frame is not None:
                        frame_copy = self.current_frame.copy()
                
                try:
                    # Run analysis
                    from PIL import Image
                    import tempfile
                    
                    # Save frame temporarily
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        Image.fromarray(frame_copy).save(tmp.name)
                        results = self.pipeline.process_image(tmp.name)
                    
                    self.analysis_results = results
                    
                    # Calculate analysis FPS
                    analysis_count += 1
                    elapsed = current_time - last_analysis
                    self.analysis_fps = 1.0 / elapsed
                    
                except Exception as e:
                    print(f"Analysis error: {e}")
                
                last_analysis = current_time
            
            time.sleep(0.1)
    
    def _display_loop(self):
        """Display frames with overlays"""
        cv2.namedWindow('Live Preview', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Live Preview', 1280, 720)
        
        while self.is_running:
            with self.frame_lock:
                if self.current_frame is None:
                    continue
                
                display_frame = self.current_frame.copy()
            
            # Convert RGB to BGR for OpenCV
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            
            # Draw analysis overlay
            if self.analysis_results:
                display_frame = self._draw_analysis_overlay(
                    display_frame,
                    self.analysis_results
                )
            
            # Draw info overlay
            self._draw_info_overlay(display_frame)
            
            # Show frame
            cv2.imshow('Live Preview', display_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop()
            elif key == ord('c'):
                # Capture high-res image
                self._capture_still()
            elif key == ord('a'):
                # Toggle analysis
                pass
        
        cv2.destroyAllWindows()
    
    def _draw_analysis_overlay(self, frame, results):
        """Draw detection boxes and counts on frame"""
        # Draw boxes
        for box, label in zip(results['boxes'], results['labels']):
            x1, y1, x2, y2 = box.astype(int)
            color = (0, 255, 0) if label == 0 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw count
        text = f"Cells: {results['total_cells']} | Viable: {results['viability_percent']:.1f}%"
        cv2.putText(frame, text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def _draw_info_overlay(self, frame):
        """Draw FPS and other info"""
        info_text = f"FPS: {self.fps:.1f} | Analysis: {self.analysis_fps:.1f}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Q: Quit | C: Capture | A: Toggle Analysis", 
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _capture_still(self):
        """Capture full-resolution image"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        
        with self.frame_lock:
            if self.current_frame is not None:
                from PIL import Image
                Image.fromarray(self.current_frame).save(filename)
                print(f"Saved: {filename}")

# Usage
from camera_factory import create_camera
from inference.pipeline import CellCountingPipeline

camera = create_camera('opencv', '0')
camera.open()

# Load models
pipeline = CellCountingPipeline(...)

# Start live preview
preview = LivePreview(camera, pipeline)
preview.start(enable_analysis=True, analysis_interval=2.0)
```

**Deliverable**: Live preview with real-time analysis

---

### Task 5.3: Auto-Focus Implementation (Day 7-9)
**Goal**: Automatically adjust focus for optimal image quality

**Focus Metrics**:
```python
def calculate_focus_measure(image: np.ndarray, method='laplacian') -> float:
    """
    Calculate image sharpness/focus quality
    
    Methods:
        - laplacian: Variance of Laplacian (common, fast)
        - gradient: Sum of gradient magnitudes
        - sobel: Sobel operator variance
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if method == 'laplacian':
        # Variance of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        focus = laplacian.var()
    
    elif method == 'gradient':
        # Gradient magnitude
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(gx**2 + gy**2)
        focus = gradient.sum()
    
    elif method == 'sobel':
        # Sobel variance
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        focus = sobel.var()
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return focus
```

**Auto-Focus Algorithm**:
```python
class AutoFocus:
    """
    Automatic focus adjustment for motorized microscopes
    """
    
    def __init__(self, camera: CameraInterface, focus_controller):
        """
        Args:
            camera: Camera interface
            focus_controller: Object with set_focus(position) method
        """
        self.camera = camera
        self.focus_controller = focus_controller
    
    def find_optimal_focus(self, 
                          method='hill_climbing',
                          focus_range=None,
                          step_size=10):
        """
        Find optimal focus position
        
        Args:
            method: 'hill_climbing', 'coarse_fine', or 'adaptive'
            focus_range: (min, max) focus positions
            step_size: Initial step size
        
        Returns:
            Optimal focus position
        """
        if method == 'hill_climbing':
            return self._hill_climbing(focus_range, step_size)
        elif method == 'coarse_fine':
            return self._coarse_fine_search(focus_range)
        elif method == 'adaptive':
            return self._adaptive_search(focus_range)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _hill_climbing(self, focus_range, step_size):
        """
        Hill climbing algorithm for focus
        """
        if focus_range is None:
            focus_range = (0, 1000)
        
        min_pos, max_pos = focus_range
        current_pos = (min_pos + max_pos) // 2
        
        # Set initial position
        self.focus_controller.set_focus(current_pos)
        time.sleep(0.5)  # Wait for focus to settle
        
        # Get initial focus measure
        image = self.camera.capture_image()
        current_focus = calculate_focus_measure(image)
        
        direction = 1  # Start moving up
        consecutive_decreases = 0
        
        while step_size > 1:
            # Try next position
            next_pos = current_pos + (direction * step_size)
            
            # Check bounds
            if next_pos < min_pos or next_pos > max_pos:
                direction *= -1  # Reverse direction
                step_size //= 2  # Reduce step
                continue
            
            # Move to position
            self.focus_controller.set_focus(next_pos)
            time.sleep(0.3)
            
            # Measure focus
            image = self.camera.capture_image()
            next_focus = calculate_focus_measure(image)
            
            print(f"Position: {next_pos}, Focus: {next_focus:.2f}")
            
            if next_focus > current_focus:
                # Improvement found
                current_pos = next_pos
                current_focus = next_focus
                consecutive_decreases = 0
            else:
                # No improvement
                consecutive_decreases += 1
                
                if consecutive_decreases >= 2:
                    # Change direction and reduce step
                    direction *= -1
                    step_size //= 2
                    consecutive_decreases = 0
        
        print(f"Optimal focus: {current_pos}, measure: {current_focus:.2f}")
        return current_pos
    
    def _coarse_fine_search(self, focus_range):
        """
        Two-stage: coarse sweep then fine adjustment
        """
        min_pos, max_pos = focus_range
        
        # Coarse sweep
        coarse_step = (max_pos - min_pos) // 10
        best_pos = min_pos
        best_focus = 0
        
        print("Coarse search...")
        for pos in range(min_pos, max_pos, coarse_step):
            self.focus_controller.set_focus(pos)
            time.sleep(0.3)
            
            image = self.camera.capture_image()
            focus = calculate_focus_measure(image)
            
            print(f"  Position: {pos}, Focus: {focus:.2f}")
            
            if focus > best_focus:
                best_focus = focus
                best_pos = pos
        
        # Fine adjustment around best position
        print("Fine search...")
        fine_min = max(min_pos, best_pos - coarse_step)
        fine_max = min(max_pos, best_pos + coarse_step)
        
        return self._hill_climbing((fine_min, fine_max), step_size=5)

# Usage (if motorized focus available)
# autofocus = AutoFocus(camera, focus_controller)
# optimal_focus = autofocus.find_optimal_focus(
#     focus_range=(0, 1000),
#     method='coarse_fine'
# )
```

**Deliverable**: Auto-focus implementation

---

### Task 5.4: Batch Capture & Time-Lapse (Day 10-11)
**Goal**: Automated multi-image acquisition

**Batch Capture**: `microscope/batch_capture.py`
```python
class BatchCapture:
    """
    Capture multiple images with automated processing
    """
    
    def __init__(self, camera, pipeline, output_dir='batch_results'):
        self.camera = camera
        self.pipeline = pipeline
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def capture_batch(self, 
                     num_images: int,
                     interval: float = 1.0,
                     auto_analyze: bool = True):
        """
        Capture series of images
        
        Args:
            num_images: Number of images to capture
            interval: Seconds between captures
            auto_analyze: Run analysis on each image
        """
        results = []
        
        print(f"Starting batch capture: {num_images} images")
        
        for i in range(num_images):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"batch_{timestamp}_{i:03d}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # Capture image
            print(f"Capturing {i+1}/{num_images}...")
            image = self.camera.capture_image()
            
            # Save
            from PIL import Image
            Image.fromarray(image).save(filepath)
            
            # Analyze
            if auto_analyze:
                analysis = self.pipeline.process_image(filepath)
                results.append({
                    'filename': filename,
                    'total_cells': analysis['total_cells'],
                    'viable_cells': analysis['viable_cells'],
                    'viability_percent': analysis['viability_percent']
                })
                print(f"  Cells: {analysis['total_cells']}, "
                      f"Viability: {analysis['viability_percent']:.1f}%")
            
            # Wait for interval
            if i < num_images - 1:
                time.sleep(interval)
        
        # Save results
        if auto_analyze:
            self._save_batch_results(results)
        
        print("Batch capture complete!")
        return results
    
    def _save_batch_results(self, results):
        """Save results to CSV"""
        import pandas as pd
        
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, 'batch_results.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved: {csv_path}")
    
    def time_lapse(self,
                   duration_minutes: float,
                   interval_seconds: float,
                   auto_analyze: bool = True):
        """
        Time-lapse image capture
        
        Args:
            duration_minutes: Total duration
            interval_seconds: Time between captures
            auto_analyze: Run analysis on each image
        """
        num_images = int((duration_minutes * 60) / interval_seconds)
        
        print(f"Time-lapse: {num_images} images over {duration_minutes} minutes")
        
        return self.capture_batch(
            num_images=num_images,
            interval=interval_seconds,
            auto_analyze=auto_analyze
        )

# Usage
batch = BatchCapture(camera, pipeline)

# Capture 10 images, 30s apart
batch.capture_batch(num_images=10, interval=30.0)

# Or time-lapse: 2 hours, 5 min intervals
batch.time_lapse(duration_minutes=120, interval_seconds=300)
```

**Deliverable**: Batch and time-lapse capabilities

---

### Task 5.5: Integration with Web App (Day 12-13)
**Goal**: Add camera features to web interface

**Backend Updates**: Add to `backend/main.py`
```python
from microscope.camera_factory import create_camera
from microscope.live_preview import LivePreview

# Global camera instance
camera = None

@app.on_event("startup")
async def startup_camera():
    """Initialize camera if available"""
    global camera
    try:
        camera = create_camera('opencv', '0')
        camera.open()
        print("Camera initialized")
    except Exception as e:
        print(f"Camera not available: {e}")

@app.get("/api/camera/status")
async def camera_status():
    """Check if camera is available"""
    return {
        "available": camera is not None and camera.is_open,
        "resolution": camera.get_resolution() if camera else None
    }

@app.post("/api/camera/capture")
async def capture_from_camera():
    """Capture image from live camera"""
    if not camera or not camera.is_open:
        raise HTTPException(503, "Camera not available")
    
    try:
        # Capture image
        image = camera.capture_image()
        
        # Save temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            from PIL import Image
            Image.fromarray(image).save(tmp.name)
            
            # Analyze
            results = pipeline.process_image(tmp.name)
        
        return JSONResponse({
            "status": "success",
            "results": {
                "total_cells": int(results['total_cells']),
                "viable_cells": int(results['viable_cells']),
                "viability_percent": float(results['viability_percent'])
            }
        })
    
    except Exception as e:
        raise HTTPException(500, f"Capture failed: {str(e)}")

@app.get("/api/camera/stream")
async def camera_stream():
    """Stream camera feed (simplified)"""
    # For full video streaming, would need WebRTC or MJPEG
    # This is a simplified single-frame endpoint
    
    if not camera or not camera.is_open:
        raise HTTPException(503, "Camera not available")
    
    image = camera.get_live_frame()
    
    # Convert to JPEG
    from PIL import Image
    import io
    
    img_pil = Image.fromarray(image)
    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG')
    buf.seek(0)
    
    return Response(content=buf.read(), media_type="image/jpeg")
```

**Frontend Updates**: Add camera controls
```html
<!-- Add to frontend/index.html -->
<section id="cameraSection" class="bg-white rounded-lg shadow-md p-6 mb-6 hidden">
    <h2 class="text-2xl font-semibold mb-4">Live Camera</h2>
    
    <div class="grid grid-cols-2 gap-4">
        <div>
            <img id="cameraFeed" class="w-full rounded-lg border">
        </div>
        <div>
            <button id="captureBtn" class="w-full bg-green-600 text-white px-6 py-3 rounded-lg mb-4">
                📸 Capture & Analyze
            </button>
            <button id="startStreamBtn" class="w-full bg-blue-600 text-white px-6 py-3 rounded-lg">
                ▶️ Start Live Stream
            </button>
        </div>
    </div>
</section>
```

**Deliverable**: Camera features in web interface

---

### Task 5.6: Documentation & Examples (Day 14)
**Goal**: Complete hardware integration guide

**Hardware Setup Guide**: `docs/CAMERA_SETUP.md`
```markdown
# Camera Setup Guide

## Supported Cameras

### USB Webcams
- Any UVC-compatible webcam
- Plug and play, no drivers needed
- Limited control over exposure/gain

**Setup:**
1. Connect camera via USB
2. Camera will appear as `/dev/video0` (Linux) or device 0 (Windows)
3. Test: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

### Industrial Cameras

**Basler GigE/USB3:**
1. Install Pylon SDK from Basler website
2. `pip install pypylon`
3. Find camera: `pypylon-info`
4. Use serial number in config

**FLIR (formerly Point Grey):**
1. Install Spinnaker SDK
2. `pip install PySpin`
3. Configure as per FLIR docs

## Configuration

Edit `config/camera_config.yaml`:
```yaml
camera:
  type: opencv  # or basler, flir
  device_id: 0
  resolution:
    width: 1920
    height: 1080
  settings:
    exposure: auto  # or value in ms
    gain: 50
    white_balance: auto
```

## Troubleshooting

**Camera not detected:**
- Check USB connection
- Check permissions (Linux: add user to `video` group)
- Try different USB port
- Verify with manufacturer's software

**Poor image quality:**
- Adjust exposure and gain
- Check microscope illumination
- Clean camera lens
- Run auto-exposure calibration

**Slow performance:**
- Reduce resolution
- Use USB 3.0 port
- Disable auto-processing features
```

**Deliverable**: Complete camera documentation

---

## Phase 5 Success Criteria
- [ ] Camera abstraction layer supports 2+ camera types
- [ ] Live preview working with analysis overlay
- [ ] Auto-focus implemented (if hardware supports)
- [ ] Batch capture functional
- [ ] Web interface camera integration
- [ ] Documentation complete

## Limitations & Future Work
- Full video streaming requires WebRTC
- Auto-focus needs motorized stage
- Multi-position scanning needs XY stage
- Deep integration requires vendor SDKs
- Performance optimization for high-speed capture

## Integration Checklist
- [ ] Test with available camera hardware
- [ ] Verify microscope compatibility
- [ ] Calibrate camera settings for cell imaging
- [ ] Document specific hardware setup
- [ ] Create user guide for lab personnel
- [ ] Test workflow end-to-end

---

# Project Complete!

This completes the 5-phase development plan. You now have:
1. ✅ Annotated training dataset
2. ✅ Trained ML models (detection + classification)
3. ✅ Optimized, evaluated models
4. ✅ Web application interface
5. ✅ Microscope camera integration framework

**Next steps:**
- Deploy to production environment
- Collect user feedback
- Iterate on model performance
- Add advanced features (multi-well plates, etc.)
- Consider publishing as open-source tool
