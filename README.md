# SMART VIEW AI

**Transforming Vision into Intelligent Audio Guidance**

Bachelor of Computer Science graduation project · Jinan University, Lebanon · **Final academic grade: 92/100**

SMART VIEW AI is an AI-powered academic assistive technology prototype that investigates how computer vision and object detection can be translated into selective spoken descriptions for people with visual impairments. Its central engineering focus is deciding **which object to announce, when to announce it, and how to avoid repeated alerts**, rather than reading every detection aloud.

The repository brings together the original Python processing application, a Raspberry Pi audio receiver, a graduation presentation, demonstration recordings, and prototype images. It documents a research and engineering prototype; the materials do not establish a validated navigation aid or independent mobility solution.

[Project presentation](smartview%20ai%20v4%20final.pptx) · [Demo 1](videos/demo_1.mp4) · [Demo 2](videos/demo_2.mp4) · [Processing code](main.py) · [Raspberry Pi code](raspberry_pi.py)

## Project overview and motivation

Camera scenes can contain many detections competing for a user's attention. Announcing each one can produce repetitive or unnecessary speech. The project investigates selective spoken information through object detection, object tracking, configured intelligent prioritization, left/right spatial audio guidance, and embedded system integration.

The engineering objective is to select useful object descriptions while limiting competing announcements. The presentation frames this work around accessibility, humanitarian assistance, and ethical technology rather than commercial exploitation. These motivations do not establish measured safety or usability benefits.

## Objectives

- Convert camera observations into concise object-and-direction phrases, such as “person on the right.”
- Reduce competing announcements using semantic priority classes and bounding-box area filtering.
- Use tracking IDs and temporal label accumulation to suppress repeated announcements.
- Separate vision processing from speech playback through a local-network Raspberry Pi audio node.
- Develop a Smart Cane obstacle-alert prototype alongside the vision application, and explore currency recognition as an experimental extension.

## Implementation status

| Component | Status and evidence |
| --- | --- |
| General object detection and tracking | Implemented in `main.py`; a supplied screenshot shows person/chair labels, IDs, zones, and area filtering. |
| Priority selection and speech suppression | Implemented in source; no comparative user study or quantitative reduction in alerts is provided. |
| Raspberry Pi audio delivery | TCP sender and eSpeak receiver are implemented; no controlled end-to-end hardware validation report is included. |
| Money detection | Experimental. A mode and model-loading path exist, and an image shows a banknote detection. Custom weights, training data, and evaluation results are absent. |
| Smart Cane | Implemented, tested, and demonstrated obstacle-detection and alert prototype, as confirmed by the project team. A supplied photograph documents the physical prototype; quantitative test records and wiring are not included. |
| Cane sensor and alert components | Exact ultrasonic sensor, vibration motor, and buzzer models, specifications, wiring, and individual operating behavior cannot be verified from the available repository materials. |
| Mobile deployment, reading, smart glasses, haptics, offline operation | Future directions in the presentation. A phone interface appears in the videos, but its source and integration are not included in the supplied Python implementation. |

## System architecture

```mermaid
flowchart LR
    subgraph PC[Processing computer]
        A[HTTP camera stream] --> B[OpenCV frame capture]
        B --> C[90-degree clockwise rotation]
        C --> D[Frame queue]
        D --> E[YOLOv8x detection]
        E --> F[BoT-SORT tracking]
        F --> G[Area filtering and temporal confirmation]
        G --> H[Priority selection and cooldowns]
        H --> I[Speech queue]
        I --> J[TCP client]
        G --> K[Result queue]
        K --> L[Annotated OpenCV display]
    end
    subgraph PI[Raspberry Pi audio node]
        M[TCP server and line buffer] --> N[eSpeak text-to-speech]
        N --> O[Audio output]
    end
    J -->|Newline-delimited text on port 5014| M
```

This diagram shows general mode; experimental money mode substitutes the custom model when available. Area filtering occurs before application-level label-history updates and temporal confirmation. Only confirmed, eligible candidates enter priority selection, while visual annotations are produced independently of whether speech is enqueued.

Computer vision runs on the main computer. The Raspberry Pi is a separate audio processing/output node connected over the network: it receives text rather than an audio stream and synthesizes speech locally. PyTorch selects CUDA when available and otherwise uses the CPU.

### Concurrent processing

| Execution context | Role |
| --- | --- |
| Camera capture thread | Captures HTTP camera frames, rotates them 90° clockwise, and replaces an older queued frame when necessary. |
| YOLO inference thread | Processes every third dequeued frame, performs detection/tracking and speech selection, and publishes annotated frames. |
| Network sender thread | Consumes text phrases and sends them to the Pi when connected. |
| Main graphical display loop | Displays annotated results and handles keyboard mode changes and exit. |

All three queues use `maxsize=1`. `frame_queue` holds an input frame; `result_queue` holds an annotated output frame, replacing older queued results when full; `speech_queue` holds a pending phrase. Speech is enqueued without blocking and a full queue causes that enqueue attempt to be skipped. This bounds queued work and attempts to limit processing delays without guaranteeing real-time performance or a measured latency reduction.

A `threading.Lock` named `mode_lock` coordinates graphical-loop mode updates with the inference thread's model/parameter selection. Application tracking history and class cooldowns are cleared when inference observes a mode change; this is separate from the tracker's internal state.

## Computer vision pipeline and decision policy

### Object detection and tracking

General mode loads **YOLOv8x** (`yolov8x.pt`) through **Ultralytics YOLO**, with **PyTorch** for inference and **OpenCV** for capture, annotation, and display. Ultralytics uses **BoT-SORT**, configured through `botsort.yaml`, with `persist=True` to associate detections across successive inference calls and expose tracking IDs.

Detection supplies bounding boxes and class labels. Tracking associates observations with IDs over time. The application's separate `track_history` stores label counts, last qualifying observation times, and whether each ID has already been selected for speech. Detections without available tracking IDs are skipped by this application's annotation/selection loop.

These are application-level integration and decision-policy contributions built on existing YOLO and BoT-SORT components. The team did not develop a new YOLO architecture or tracking algorithm.

### Intelligent priority engine

The code selects one eligible target per inference result using the following dictionary:

| Priority | Labels configured in the source |
| --- | --- |
| 1 | `person`, `car`, `motorcycle`, `bus`, `truck`, `bicycle`, `stop sign`, `traffic light`, `fire hydrant` |
| 2 | `chair`, `couch`, `bed`, `dining table`, `bench`, `door` |
| 3 | Any other detected label |

These priority categories are configured rules, not learned risk scores. Lower numbers take precedence; equal-priority candidates are compared by bounding-box area, favoring the larger box. Money mode assigns all candidates priority 1. Priority labels are configuration entries: adding `door` to the dictionary does not make the loaded general model capable of detecting doors.

### Object filtering

| Mode | Confidence threshold | Inference image size (`imgsz`) | Minimum box area |
| --- | --- | --- | --- |
| General | 0.60 | 160 | 18,000 pixels² |
| Experimental money | 0.60 | 480 | 4,000 pixels² |

Bounding-box area is `(x2 - x1) * (y2 - y1)`. These are configured parameters, not measured performance results. Area is computed from box coordinates in the captured frame.

Boxes below the area threshold remain visible with a white outline and a “Too Far” label but are excluded from spoken-target selection. This is an image-area heuristic: apparent size depends on object dimensions, framing, and camera geometry. It is **not a calibrated distance measurement** and is separate from ultrasonic ranging.

### Temporal detection confirmation and speech suppression

The application maintains label counts per tracking ID and selects the most frequently observed label for that ID. An object becomes eligible after that label accumulates at least **five qualifying detections** (`REQUIRED_FRAMES = 5`); these need not be consecutive frames. Only tracked detections passing the area threshold update these counts. This application-level temporal filter aims to reduce speech from brief detections, improve spoken-label consistency, and limit unnecessary output; it does not guarantee correct identification or have a controlled quantitative effectiveness evaluation.

Once a phrase is enqueued, the track is marked as spoken. A class-level cooldown suppresses the same label for **five seconds**, while a global speech interval of **3.5 seconds** separates enqueued phrases. Track history expires after five seconds without a qualifying observation, and application history is cleared on a mode change. These mechanisms aim to reduce repetition; tracking errors and reidentification can still lead to repeated alerts. Enqueueing a phrase does not confirm successful delivery or audible playback.

### Left/right spatial guidance

The horizontal box center is conceptually `cx = (x1 + x2) / 2`; the implementation uses integer division, `cx = (x1 + x2) // 2`, and compares it with `frame_width // 2`. A center to the left of that midpoint produces “left”; all other positions produce “right.” Example phrases are “person on the left” and “chair on the right.”

There are exactly two horizontal zones, with no center zone. These image-relative descriptions provide no GPS navigation, route planning, calibrated physical distance, or collision prediction.

## TCP communication and Raspberry Pi speech

The processing computer is an IPv4 TCP client (`AF_INET`, `SOCK_STREAM`); the Pi is the TCP server on port **5014**. The message protocol is UTF-8 text terminated by a newline character.

1. YOLO detects objects and BoT-SORT supplies tracking IDs.
2. The decision policy selects an eligible object and generates a class-and-direction phrase.
3. The phrase enters `speech_queue`.
4. The network sender transmits the phrase with its newline delimiter.
5. The Pi receives data and accumulates it in a persistent text buffer.
6. Complete newline-delimited messages are extracted and passed to eSpeak.
7. eSpeak generates speech locally for the configured Pi audio output.

For each completed nonempty phrase, the receiver invokes:

```text
espeak -s 150 -g 10 <received phrase>
```

The persistent buffer retains incomplete lines across receives and can extract multiple lines from one receive. The receiver decodes each received chunk separately with invalid UTF-8 bytes ignored; it is not a full incremental Unicode decoder.

`raspberry_pi.py` is written in Python, uses the standard-library `socket` and `subprocess` modules, and invokes eSpeak synchronously with speech parameters `-s 150` and `-g 10`. Thonny was the Pi development environment: it is an IDE, not a programming language. A GitHub language breakdown of 100% Python is therefore consistent with the repository source files. The exact audio output hardware is not established by the supplied materials.

The Pi listens on all interfaces on TCP port `5014` and handles one client at a time. Speech execution is synchronous. The client makes one initial connection attempt and does not automatically reconnect. The protocol has no authentication, encryption, delivery acknowledgement, or audio-completion feedback; use a trusted local network.

## Observed results

### General detection screenshot

![General detection with person and chair tracking IDs, left/right zones, and an area-filtered box](WhatsApp%20Image%202026-11-12%20at%2010.41.18%20AM.jpeg)

The supplied screenshot shows a person and chair with tracking IDs, accumulated label counts, the center line, and a white “Too Far” box. It supports the presence of visual detection and filtering output for this scene. Its instantaneous FPS overlay is not a benchmark, average throughput, or end-to-end latency measurement. A still image cannot establish ID stability over time, speech quality, or safe navigation.

## Experimental money detection

![Experimental money detection screen showing a US twenty-dollar banknote](WhatsApp%20Image%202026-11-12%20at%2011.06.00%20AM.jpeg)

The project team reports developing and testing an experimental money detection feature using a custom YOLO-based object detection model and a dataset prepared through Roboflow. Dataset size, collection methodology, training work, and qualitative outcomes below are project-reported experimental information; the underlying artifacts are not included for independent verification.

The objective was to explore whether the system could recognize banknotes and provide audio feedback to visually impaired users.

### Dataset development

The team reports creating a custom image dataset containing four currency categories.

* Dataset Size: Approximately 400 images in total.
* Number of Classes: 4 currency categories.
* Images per Class: Approximately 100 images.
* Dataset Platform: Roboflow.
* Image Diversity: Banknotes were photographed from multiple viewing angles, orientations, and positions.
* Background Variation: Different backgrounds were used to introduce visual diversity and help the model learn under varying conditions.

Rather than relying exclusively on a pretrained general-purpose detection model, this experiment involved collecting task-specific images and preparing a dedicated dataset for currency recognition.

### Custom model training

The collected dataset was managed using Roboflow and used to train a custom YOLO-based object detection model.

The experiment involved:

* Collecting and organizing images into four categories.
* Preparing a custom dataset for object detection.
* Training a task-specific detection model.
* Testing the model’s ability to identify banknotes.
* Observing recognition limitations under different visual conditions.

This process provided practical experience in custom dataset development, computer vision workflows, and object detection model training.

### Experimental integration

An experimental money detection mode was implemented within the SMART VIEW AI application.

At startup, the source attempts to load custom YOLO weights from `best.pt`; selecting money mode later chooses the model already loaded.

The supplied experimental screenshot includes an example of the application operating in money detection mode, with a bounding box around a US twenty-dollar banknote.

However, the current GitHub repository does not include the trained model weights, original dataset, training configuration, complete class list, or independent evaluation results.

If `best.pt` cannot be loaded, the application prints an error and falls back to the general object detection model.

Therefore, selecting money detection mode does not necessarily confirm that the custom currency recognition model is active.

### Audio feedback

The experimental money detection mode uses the application’s existing object announcement mechanism.

Detected objects are processed through the same class-and-direction speech template used in the general detection mode.

A separate currency-specific speech formatting system was not implemented.

### Observations and limitations

Although the custom model was trained and the feature was explored during development, its performance was not sufficiently reliable for inclusion as a validated component of the final assistive system.

The available demonstration does not establish reliable recognition across different denominations, currencies, lighting conditions, backgrounds, or viewing angles.

Additionally, verified quantitative evaluation metrics, such as precision, recall, and mean Average Precision (mAP), are not available in this repository.

For these reasons, money detection remained an experimental feature rather than a fully deployed currency recognition system.

### Technical experience gained

This experiment provided hands-on experience in:

* Custom dataset collection and organization.
* Dataset management using Roboflow.
* Image diversity and background variation.
* Training custom YOLO-based object detection models.
* Integrating experimental AI models into an existing application.
* Understanding the importance of model evaluation and real-world reliability.

Although money detection was not retained as a validated feature of the final system, the experiment contributed to the project’s exploration of task-specific computer vision applications.

## Smart cane prototype and ultrasonic sensing

![Physical smart cane prototype demonstrated indoors alongside a handheld phone](WhatsApp%20Image%202026-11-12%20at%2011.11.37%20AM.jpeg)

The Smart Cane is an **implemented and tested prototype**, not merely a proposed feature. The project team confirms that it was successfully built, tested, and demonstrated, detecting obstacles and providing alerts in practical use. The supplied photograph documents the physical prototype. This qualitative team report does not establish detection range, accuracy, reliability, or validation across environments.

An ultrasonic sensor estimates range from an emitted pulse's round-trip echo time. This is distinct from the vision application's bounding-box area heuristic. The available presentation text, source code, and prototype image do not identify exact ultrasonic sensor, vibration motor, or buzzer models or specifications. Sensor firmware, wiring, trigger distances, and the individual vibration/buzzer alert behavior are not documented, so no component model or operating threshold is asserted here.

Neither supplied Python application reads GPIO or ultrasonic measurements. The implemented cane should therefore be distinguished from its connection to the vision/audio pipeline: that connection is not established by the available code. The team's confirmation supports the cane's practical prototype status without implying that the complete combined system has undergone controlled hardware validation.

## Project demonstration

- [Watch Demo 1](videos/demo_1.mp4)
- [Watch Demo 2](videos/demo_2.mp4)

Both original MP4 files are preserved. Inspection here covered sampled visual frames, not a complete audiovisual review. Sampled frames show a phone interface and camera imagery; interface controls alone do not verify working reading, currency recognition, offline inference, or integration with the supplied Python programs. These recordings are qualitative project materials rather than controlled experiments.

## Academic demonstration and evaluation

According to the project team, SMART VIEW AI was successfully demonstrated during the final graduation project evaluation at Jinan University, Lebanon. The following demonstration outcomes and grade are project-reported academic information; an official evaluation record is not included in the repository.

During the live demonstration, the system successfully performed its core functions:

* Object Detection: Identified objects and people using the YOLO-based computer vision pipeline.
* Object Tracking: Tracked detected objects to support consistent identification.
* Intelligent Prioritization: Applied filtering and priority rules to determine which detected objects should be announced.
* Audio Guidance: Generated spoken feedback describing detected objects and their relative positions.
* System Integration: Demonstrated the operation of the computer vision pipeline and its audio feedback components.

The live demonstration confirmed that the core system components could operate together successfully in a controlled academic environment.

**Final graduation project grade: 92/100**

The project was evaluated as part of the Bachelor’s degree in Computer Science at Jinan University.

Although the live demonstration was successful, it does not establish validated performance in uncontrolled real-world environments.

Further testing would be required to evaluate detection accuracy, system latency, navigation reliability, and usability with visually impaired individuals.

## Technologies and dependencies

| Category | Technology and role |
| --- | --- |
| Programming — Python | Processing application and Pi audio receiver |
| Computer vision — OpenCV (`opencv-python`) | HTTP camera capture, rotation, annotation, graphical display |
| Artificial intelligence — YOLOv8x / PyTorch (`torch`) | CPU/CUDA inference support |
| AI framework — Ultralytics (`ultralytics`) | YOLO model loading and tracking API |
| Tracking — BoT-SORT | Tracking through Ultralytics' `botsort.yaml` configuration |
| Networking — TCP/IP and Python sockets | IPv4 TCP text transport using the standard library |
| Concurrency — Python threading and queues | Capture, inference, and network workers; bounded queues and mode lock |
| Embedded computing — Raspberry Pi | Separate network audio node |
| Text-to-speech — eSpeak | Local speech synthesis on the Pi |
| Development environment — Thonny IDE | Development/run environment for the Pi application; not a runtime dependency |
| Experimental dataset development — Roboflow | Custom currency-model development tool reported in the presentation; training artifacts are not included |

The third-party Python imports are listed in [requirements.txt](requirements.txt). The Pi receiver uses only `socket` and `subprocess` from the standard library, plus the external eSpeak executable. Original package and Python versions were not supplied, so dependencies are unpinned. Some Ultralytics versions may require additional tracking dependencies at runtime.

## Installation and operation

### Processing computer

Use a Python version supported by the installed PyTorch, Ultralytics, and OpenCV releases. A graphical desktop is required for `cv2.imshow`.

```bash
git clone https://github.com/omarsaadeh/SMART-VIEW-AI.git
cd SMART-VIEW-AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. For CUDA, follow the official PyTorch installation instructions for a build compatible with the local GPU and drivers. CPU fallback is implemented, but throughput is hardware-dependent.

Run from the repository root. Supply `yolov8x.pt` there, or allow Ultralytics to download it. To reproduce the currency experiment, obtain the original custom `best.pt` separately; a general YOLO model is not a replacement for those weights. Model files are excluded by `.gitignore`.

### Raspberry Pi audio node

On Raspberry Pi OS, install eSpeak and configure a working audio output:

```bash
sudo apt update
sudo apt install espeak
python3 raspberry_pi.py
```

Alternatively, open `raspberry_pi.py` in Thonny and run it on the Pi. The computer's vision dependencies are not required for this audio-only script.

### Network settings and launch order

The original source values are preserved:

| Setting | Current value |
| --- | --- |
| Raspberry Pi address in `main.py` | `172.20.10.8` |
| TCP port in both scripts | `5014` |
| HTTP camera URL in `main.py` | `http://172.20.10.2:8080/video` |

For a different deployment, these source settings must match the actual devices; the programs do not load environment variables or command-line configuration. No source settings were changed for this portfolio update.

1. Start an HTTP camera stream reachable by the processing computer.
2. Start the Pi audio receiver before the processing application.
3. Activate the processing environment and run `python main.py` from the repository root.
4. Focus the OpenCV window: `1` selects general detection, `2` selects experimental money mode, and `q` exits when the display loop is receiving frames.
5. Stop the Pi receiver with `Ctrl+C`.

## Technical limitations

- **Validation:** No comprehensive real-world safety or usability study with visually impaired individuals, or quantitative accuracy, precision, recall, mAP, latency, throughput, or tracking evaluation is supplied. A successful academic demonstration establishes prototype integration, not deployment readiness.
- **Vision and spatial guidance:** Bounding-box area is not calibrated depth estimation. Guidance has only left/right zones. Lighting, darkness, occlusion, crowding, camera framing, and model label coverage can affect detection; the presentation identifies darkness and extreme crowding as limitations, without measured operating boundaries.
- **Speech decisions:** Tracking errors, ID changes, and accumulated label errors can produce missed, incorrect, or repeated announcements. Speech eligibility and enqueueing do not establish delivery or playback.
- **Networking:** A reachable local network is required by the present setup. There is no automatic client reconnection, end-to-end delivery acknowledgement, or audio-completion feedback. Disconnected-client phrases can be consumed without transmission.
- **Currency recognition:** Custom weights, training data/configuration, complete class list, and evaluation artifacts are absent. The fallback general model does not reproduce the experiment.
- **Hardware and reproducibility:** Some cane components, wiring, thresholds, audio hardware, and hardware test records are undocumented. Dependencies are unpinned, original environment versions are unavailable, and the supplied Python scripts do not establish cane/vision integration.

## Future improvements

Future work includes systematic environmental and user testing; quantitative detection, tracking, and speech-delivery evaluation; processing optimization and improved announcement policies; better evaluated currency recognition; and more extensive integration of the demonstrated Smart Cane with the vision system.

Mobile deployment, smart glasses, expanded spatial guidance, reading mode, haptic feedback, and fully offline operation are research directions rather than established features of the supplied implementation. Local vision inference already occurs on the computer, but the current camera and Pi architecture still uses a network.

## Team and verified contributions

The presentation identifies the team and supervisor. The following full names, roles, and individual contributions were subsequently confirmed by the project team:

| Person | Verified attribution |
| --- | --- |
| Omar Ahmad Saadeh — Team Leader | Led and coordinated the team; responsible for project planning and technical decision-making; worked on main software development and AI system integration. |
| Ali Abdelwahed | Developed and assembled the Smart Cane; assisted with Raspberry Pi software development and integration. |
| Wael Saadeh | Assisted with project hardware development and integration. |
| Dr. Rami Sefarjaleni | Project supervisor |

## Repository materials and evidence

| Material | Reference |
| --- | --- |
| Original processing application | [main.py](main.py) |
| Original Raspberry Pi receiver | [raspberry_pi.py](raspberry_pi.py) |
| Python dependencies | [requirements.txt](requirements.txt) |
| Graduation presentation, 12 slides | [smartview ai v4 final.pptx](smartview%20ai%20v4%20final.pptx) |
| General detection result | [Original image](WhatsApp%20Image%202026-11-12%20at%2010.41.18%20AM.jpeg) |
| Currency experiment | [Original image](WhatsApp%20Image%202026-11-12%20at%2011.06.00%20AM.jpeg) |
| Cane prototype | [Original image](WhatsApp%20Image%202026-11-12%20at%2011.11.37%20AM.jpeg) |
| Demonstration recordings | [Demo 1](videos/demo_1.mp4), [Demo 2](videos/demo_2.mp4) |

This README uses the source for software implementation details, readable slide text for project intent and attribution, supplied images for scene-specific observations, and explicit project-team confirmation for individual contributions and the Smart Cane’s built/tested/demonstrated status. Presentation aspirations are distinguished from code-backed functionality. The portfolio update preserves all original source and media files.

## Academic prototype disclaimer

SMART VIEW AI was developed as an academic graduation project to explore the application of Artificial Intelligence and Computer Vision in assistive technologies for visually impaired individuals.
The system is a proof-of-concept prototype intended for educational and research demonstration purposes.
Although the prototype was successfully demonstrated, it has not undergone the comprehensive security, reliability, and safety validation required for real-world deployment.
The project is not intended to replace certified mobility aids or professional assistance.
