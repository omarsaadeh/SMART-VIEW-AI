# SMART VIEW AI

University graduation project: an AI-powered assistive system for visually impaired people using computer vision and real-time audio guidance.

## Project files

| File | Purpose |
| --- | --- |
| `main.py` | Processing application: camera input, YOLO tracking, detection display, and guidance messages over TCP. |
| `raspberry_pi.py` | Raspberry Pi audio application: receives messages and speaks them using eSpeak. Can be run from Thonny or a terminal. |
| `requirements.txt` | Third-party Python dependencies imported by the processing application. |

The original application functionality is preserved. The Raspberry Pi code's chat formatting has been converted to normal Python indentation and comments.

## Operation

The main application reads an HTTP camera stream and rotates frames clockwise. General detection uses YOLOv8x. Money detection uses a separately supplied custom model. BoT-SORT tracks objects; area thresholds, accumulated label observations, object priorities, and speech cooldowns determine guidance. Messages describe objects on the left or right and are sent as newline-delimited text to the Raspberry Pi.

The audio application buffers TCP data until a newline arrives, then calls eSpeak for each nonempty phrase. It handles one connected client at a time.

## Processing computer setup

Use Python with compatible PyTorch, Ultralytics, and OpenCV packages. The original working Python and package versions were not supplied; dependencies are therefore unpinned.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.

A graphical desktop is required for the OpenCV display. The code selects CUDA when PyTorch reports it available, otherwise CPU. For GPU use, install a PyTorch build compatible with your hardware using the official PyTorch installation instructions.

### Model weights

Run the main application from this repository directory:

- `yolov8x.pt`: general detection weights. Ultralytics may download these if absent, requiring Internet access.
- `best.pt`: custom money detection weights, which must be supplied separately. If loading fails, the original code falls back to the general model; custom money recognition is then unavailable.

No model weights are included. `.gitignore` excludes `.pt` files and generated `runs/` output. Tracking uses the existing `botsort.yaml` setting; some Ultralytics versions may install additional tracking dependencies at runtime.

### Network configuration

The original values remain in `main.py`:

| Setting | Value |
| --- | --- |
| Raspberry Pi `RASP_IP` | `172.20.10.8` |
| TCP `PORT` | `5014` |
| Camera `url` | `http://172.20.10.2:8080/video` |

Adjust these source values to your actual devices before running on a different network. The code does not read environment variables. The Raspberry Pi uses port `5014` as well; both ports must match. The processing computer must reach the camera and Raspberry Pi. The original client attempts its connection once at startup and does not reconnect automatically.

## Raspberry Pi setup

The Pi application imports only standard-library modules, so it does not need the processing application's Python packages. On Raspberry Pi OS, install the eSpeak executable:

```bash
sudo apt update
sudo apt install espeak
```

Configure and test the Pi's audio output. Run:

```bash
python3 raspberry_pi.py
```

Alternatively, open `raspberry_pi.py` in Thonny on the Pi and run it there. The server listens on all interfaces on TCP port `5014`. Use a trusted local network: the original protocol does not authenticate or encrypt messages.

## Run the system

1. Start the HTTP camera stream.
2. Start `raspberry_pi.py` on the Raspberry Pi.
3. On the processing computer, activate the environment and run from the repository directory:

   ```bash
   python main.py
   ```

4. Focus the OpenCV window and use the controls below.

| Key | Action |
| --- | --- |
| `1` | General detection |
| `2` | Money detection, using custom weights when available |
| `q` | Quit the processing application |

Stop the Raspberry Pi node with `Ctrl+C`.

## Validation

Both scripts have passed Python syntax checks. End-to-end detection, CUDA inference, custom money recognition, and Raspberry Pi speech have not been validated during repository preparation; these require the actual devices and model weights. No automated test suite was supplied.
