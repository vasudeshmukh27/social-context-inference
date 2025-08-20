# Social Context Audio Classifier

A privacy-preserving Streamlit web app and PyTorch pipeline that classifies short audio clips into social-context categories. The app loads a trained model and predicts one of five classes: Alone / Quiet, Group Discussion, Noisy Environment, One-on-One Conversation, Speech / Monologue.


This repository contains:
- app.py — Streamlit web app for inference.
- model-training.ipynb — End-to-end training notebook that builds the dataset, extracts features, trains a CNN, evaluates it, and exports best_model.pth.
- notebook455d9c1e14.ipynb — Preprocessing and augmentation pipeline on Kaggle/GCS (AMI + MiniLibriMix).
- requirements.txt — Python dependencies for the app.

Note: The app expects a trained weights file best_model.pth in the project root. The app will gracefully notify if it’s missing.


### ✅ Requirements
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![Google Cloud](https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white)

See requirements.txt for the app:
- torch
- librosa
- numpy
- pandas
- scikit-learn
- streamlit

Training notebooks may require additional libraries (matplotlib, seaborn, soundfile, tqdm, scikit-learn). Install as needed in the notebook environment.

### ✨ Features
- 5-class audio scene classification with a lightweight CNN.
- Robust Mel Spectrogram feature extraction with librosa.
- Early-stopping training loop with best checkpoint export.
- Noise augmentation pipeline (MiniLibriMix) to create a Noisy Environment class.
- Streamlit UI with audio upload and instant predictions.


### 📦 Project Structure
- app.py
- model-training.ipynb
- notebook455d9c1e14.ipynb
- requirements.txt


### 🛠 Installation
Create and activate a virtual environment (recommended), then install dependencies:
```buildoutcfg
pip install -r requirements.txt
```

If librosa raises backend errors, also install soundfile:
```buildoutcfg
pip install soundfile
```

### 🧪 Train the Model (Kaggle/Colab friendly)

Use model-training.ipynb:
- Load the final_labeled_dataset.pkl created by preprocessing.
- Build Mel Spectrograms with audio_to_mel_spectrogram.
- Split into train/val/test with stratification.
- Train AudioClassifierCNN with early stopping.
- Automatically save the best checkpoint to best_model.pth.

The notebook prints a classification report and confusion matrix at the end, and saves best_model.pth in the working directory. Download that file and place it next to app.py.


### 🎛 Run the Web App
Ensure best_model.pth is present beside app.py, then run:
```buildoutcfg
streamlit run app.py
```

The app will:
- Warn gracefully if best_model.pth is missing.
- Let users upload wav/mp3/m4a.
- Predict one of the five classes and display the result.

### 🔊 Data Pipeline (Preprocessing and Augmentation)

notebook455d9c1e14.ipynb builds the complete dataset:
- Stage 1: Parse AMI meetings.xml, build channel→speaker map.
- Stage 2: Parse per-meeting segment XMLs, consolidate speaker timelines.
- Stage 3: Slide a 3s window with 1.5s hop across meetings; label by unique active speakers in window:
  - 0 → Alone / Quiet
  - 1 → Speech / Monologue
  - 2 → One-on-One Conversation
  - ≥3 → Group Discussion

- Extract synchronized IHM (clean, mixed headsets) and SDM (Array1-01) audio clips; pad to exact length.
- Save processed_ami_data.pkl.
- Phase 2: Augment Speech / Monologue with MiniLibriMix noise (10dB SNR) to synthesize Noisy Environment.
- Save final_labeled_dataset.pkl.

Storage-friendly notes:
- Uses gcsfs to stream audio directly from GCS (bucket: ring-ami-dataset-storage) to avoid local disk blowups.
- Processes in windows; memory footprint is controlled.
- Clearly logs counts and sample structures.

### 🧠 Model Architecture
AudioClassifierCNN (PyTorch):
- Conv2d(1→16) + BN + ReLU + MaxPool(2)
- Conv2d(16→32) + BN + ReLU + MaxPool(2)
- Conv2d(32→64) + BN + ReLU + MaxPool(2)
- Flatten
- Linear(23,552→128) + ReLU + Dropout(0.5)
- Linear(128→5)

Input: Log-Mel Spectrogram (128 mels, n_fft=1024, hop_length=256, sr=16k). With 3s clips, this yields 128×188 time-frequency cells pre-pooling.

Loss/Optimizer:
- CrossEntropyLoss
- Adam(lr=1e-3)
- Early stopping on validation loss with patience=5


### 📈 Example Results (from the included training notebook)
- Test accuracy ~0.76 with balanced performance across classes after augmentation and early stopping.
- Confusion matrix and classification report are printed at the end of training.

Results will vary with:
- The number of AMI meetings processed.
- Augmentation intensity and ratio.
- Train/val/test splits and random seeds.
- Changes to spectrogram parameters.

### 🧩 Adapting or Extending
- Add classes: Extend class_names in app.py and retrain with updated labels.
- Change features: Modify audio_to_mel_spectrogram parameters (n_mels, hop_length) and recompute in_features.
- Improve robustness: Add SpecAugment, time/frequency masking, or room impulse convolution.
- Model variants: Swap to CRNN or add attention for improved temporal modeling.

### ❗ Common Issues
- “Model not found!” in Streamlit:
  - Place best_model.pth next to app.py or update model_path in load_model().

- size mismatch in Linear:
  - If spectrogram/window settings change, recompute the Flatten size and set in_features accordingly.

- soundfile/ffmpeg errors:
  - Install soundfile; for rare codecs, convert audio to WAV before upload.

- Slow preprocessing:
  - Limit meetings, or run preprocessing on Kaggle/Colab with GCS; the pipeline is streaming-friendly.



### 🚀 Quick Start
1. Preprocess data and produce final_labeled_dataset.pkl (Kaggle/Colab).
2. Train with model-training.ipynb to produce best_model.pth.
3. Copy best_model.pth into the project root.
4. Run the app: streamlit run app.py.
5. Upload a 3s audio clip; get a predicted social context instantly.


### 🙌 Acknowledgements
- AMI Meeting Corpus (IHM/Array streams) for rich conversational audio.
- MiniLibriMix for diverse speech-in-noise augmentation.
- The open-source Python ecosystem: PyTorch, librosa, Streamlit, pandas, scikit-learn.
