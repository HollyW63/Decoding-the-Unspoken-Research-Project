# Decoding the Unspoken: Multimodal Emotion Recognition

This project combines **real-time facial emotion detection** using [DeepFace](https://github.com/serengil/deepface) with **tone-of-voice analysis** using a pretrained [Wav2Vec2 model](https://huggingface.co/superb/wav2vec2-large-superb-er) from Hugging Face.

> **Project Context:** This system analyzes the incongruence between facial expressions and vocal tones to help research social confidence in introverts. It captures data from a webcam and microphone, overlays live predictions, and automatically logs session data for statistical analysis.

---

## Features

### Core Prototype
- Real-time **face detection and emotion classification** (7 emotions)
- Real-time **voice tone analysis** (Positive/Neutral/Negative)
- Multithreaded microphone processing (no lag!)
- Combined overlay on live webcam stream
- Uses fully **pretrained** models — no training required

### Science Fair Additions (New)
- [cite_start]**Automated Data Logging:** Automatically generates timestamped `.csv` files for every session[cite: 48].
- [cite_start]**Privacy Protocol:** Assigns random Participant IDs to strictly protect user anonymity[cite: 48].
- **Sentiment Mapping:** Implements logic to map disparate emotion labels into broad **Positive vs. Negative** categories for robust correlation.
- **Analysis Notebooks:** Includes Jupyter Notebooks for generating Confusion Matrices and Linear Regression visualizations.

---

## Technologies Used

- `DeepFace` – facial expression detection
- `Wav2Vec2` from Hugging Face – tone/emotion from audio
- `PyAudio` – live microphone input
- `OpenCV` – camera display and UI overlay
- `Pandas` – data logging and CSV manipulation
- `Threading` – for parallel audio processing

---

## Installation

Create and activate a Python environment (recommended):

```bash
conda create -n emotionAI python=3.9
conda activate emotionAI

```

Install dependencies (updated for analysis tools):

```bash
pip install opencv-python deepface pyaudio numpy soundfile torch transformers pandas matplotlib scikit-learn

```

If `pyaudio` fails on macOS (Apple Silicon), install portaudio first:

```brew install portaudio
export LDFLAGS="-L/opt/homebrew/lib"
export CPPFLAGS="-I/opt/homebrew/include"
pip install pyaudio
```

---

## Running the System

### 1. Start the Real-time Detector

Run the main script to start the webcam feed and data logging:

```bash
python multimodal.py

```

**You will see:**

* Bounding boxes around detected faces
* Facial emotion label with confidence %
* Voice tone label with confidence % in the top-left corner
* *A CSV file will be automatically created in the folder.*

Press `ESC` to quit.

### 2. Run the Data Analysis

To generate charts (Confusion Matrix, Regression Lines) from your collected CSV files:

1. Open VS Code.
2. Open `emotion_correlation_analysis.ipynb`.
3. Select your Python kernel and click **Run All**.

---

## Methodology Note: Emotion Mapping

To ensure accurate statistical comparison between the Video (7 classes) and Audio (4 classes) models, this project uses the following sentiment mapping:

* **Positive:** Happy, Neutral
* **Negative:** Sad, Angry, Fear
* **Excluded:** Surprise, Disgust (due to low audio detection reliability)

---

## References

[DeepFace Github](https://github.com/serengil/deepface)

[Hugging Face Model - superb/wav2vec2-large-superb-er](https://huggingface.co/superb/wav2vec2-large-superb-er)

[PyAudio Doc](https://people.csail.mit.edu/hubert/pyaudio/)

---

## Credit

**Current Project Maintainer (GSDSEF Science Fair 2026):**

* **Holly Wang**: Responsible for experimental design, data logging implementation, privacy protocols, and statistical analysis (Pandas/Jupyter).

**Original Prototype:**

* Built by [cctz123](https://github.com/cctz123) and team for the **Wharton SF AI Leadership Final Project**.
* *The core real-time pipeline and threading logic were developed during the summer program.*
