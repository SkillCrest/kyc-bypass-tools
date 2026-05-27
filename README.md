# AI Identity & Avatar Automation System (Windows / CUDA)

Production-grade modular system to:
- Enroll & verify identity via webcam face recognition (OpenCV + `face_recognition`)
- Verify documents via OCR (EasyOCR)
- Render animated avatars (avatar engine with MediaPipe motion tracking placeholder/integration)
- Stream to a virtual camera (pyvirtualcam) for OBS / Zoom / Discord
- Control everything via a PyQt6 GUI

## Project structure
Matches the required layout.

## Prerequisites (Windows)
- Python 3.10+
- NVIDIA GPU with CUDA (for PyTorch CUDA builds)

## Install
```bat
pip install -r requirements.txt
```

## Configure
Edit `config/settings.yaml` if needed.

## Run
```bat
python main_controller.py
```

## Notes
- Virtual camera streaming is controlled via `settings.yaml` (`virtual_camera.enabled`).
- If OCR/face enrollment dependencies are missing or models are unavailable, GUI surfaces errors.

