# Decoding the Unspoken: Multimodal Emotion Recognition

A real-time system combining facial expression analysis (using [DeepFace](https://github.com/serengil/deepface)) and vocal tone analysis (using [Wav2Vec2](https://huggingface.co/superb/wav2vec2-large-superb-er)).

> **Project Context:** Engineering codebase for **GSDSEF 2026**. Analyzes face-voice incongruence to study social confidence in introverts.

---

## Features

### Core Prototype
* **Real-time Detection:** Detects 7 facial emotions and vocal sentiment simultaneously.
* **No Video Lag:** Uses multithreading to process audio in the background, keeping the video smooth.
* **Live Overlay:** Shows the AI's predictions and confidence scores right on the webcam feed.
* **Pretrained Models:** Uses powerful open-source models (DeepFace & Wav2Vec2) without needing retraining.

### My Science Fair Contributions
* **Automated Data Logging:** Scripts to automatically save every session into timestamped `.csv` files.
* **Privacy Protocol:** A system that assigns random Participant IDs (e.g., `P01`) to strictly protect anonymity.
* **Sentiment Mapping:** Logic to group different emotion labels into **Positive vs. Negative** for accurate statistical comparison.
* **Data Analysis Tools:** Custom Jupyter Notebooks to generate Confusion Matrices and Linear Regression charts from the raw data.

---

## Technologies Used

* `Python 3.9` - Primary Language
* `DeepFace` - Computer Vision
* `Wav2Vec2` - Audio Processing/Transformers
* `PyAudio` - Real-time Microphone Input
* `OpenCV` - Video Display & UI Overlay
* `Pandas` - Data Logging & Analysis

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

**⚠️ Note for macOS Users (Apple Silicon):**
If `pyaudio` fails to install, you must install `portaudio` first using Homebrew:

```bash
brew install portaudio
export LDFLAGS="-L/opt/homebrew/lib"
export CPPFLAGS="-I/opt/homebrew/include"
pip install pyaudio

```

---

## Running the System

### 1. Start the Real-time Detector

Run the main script to start the webcam feed and begin data logging:

```bash
python multimodal.py

```

**System Output:**

* Bounding boxes around detected faces.
* Facial emotion labels with confidence percentages.
* Vocal tone labels displayed in the top-left corner.
* *A new CSV file will be automatically generated in the project folder.*

Press `ESC` to quit the session.

### 2. Run the Data Analysis

To generate the charts (Confusion Matrix, Regression Lines) for the project board:

1. Open VS Code.
2. Open `emotion_correlation_analysis.ipynb`.
3. Select your Python kernel and click **Run All**.

---

## Methodology Note: Emotion Mapping

Mapping Logic: To align the 7 class Video model with the 4-class Audio model for statistical comparison:

* **Positive:** Happy, Neutral
* **Negative:** Sad, Angry, Fear
* **Excluded:** Surprise, Disgust (Excluded from correlation analysis due to lower reliability in audio detection models).

---

## References

* [DeepFace Github](https://github.com/serengil/deepface)
* [Hugging Face Model - superb/wav2vec2-large-superb-er](https://huggingface.co/superb/wav2vec2-large-superb-er)
* [PyAudio Documentation](https://people.csail.mit.edu/hubert/pyaudio/)

---

## Credits

**Current Project Maintainer (GSDSEF Science Fair 2026):**

* **Holly Wang**: Responsible for experimental design, data logging implementation, privacy protocols, and statistical analysis (Pandas/Jupyter).

**Original Prototype:**

* Built by [cctz123](https://github.com/cctz123) and team for the **Wharton SF AI Leadership Final Project**.
* *The core real-time pipeline and threading logic were developed during the summer program.*
