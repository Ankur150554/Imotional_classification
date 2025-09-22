# 🎵 Speech Emotion Recognition System

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)

### A deep learning model that analyzes audio files to classify human emotions from speech patterns. This project implements a robust Speech Emotion Recognition (SER) system using advanced feature extraction techniques and neural networks to identify emotions such as Neutral, Calm, Happy, Angry, Fearful, Disgust, and Surprised.

---

## 🎯 Project Overview

Speech Emotion Recognition (SER) is a crucial technology for understanding human emotional states through voice analysis. This project implements a complete end-to-end solution that processes audio files, extracts meaningful features, and uses a deep neural network to classify emotions with high accuracy.

The system is designed for real-world applications including customer service analysis, mental health monitoring, human-computer interaction, and educational technology platforms.

## ✨ Key Features

- **7-Class Emotion Classification**: Accurately distinguishes between Neutral, Calm, Happy, Angry, Fearful, Disgust, and Surprised emotions
- **Advanced Feature Extraction**: Utilizes multiple audio features including MFCC, Mel spectrograms, spectral features, and chroma features
- **Data Augmentation**: Implements time stretching, pitch shifting, and noise addition to improve model robustness
- **Interactive Web Interface**: User-friendly Streamlit app for real-time emotion prediction
- **Robust Preprocessing**: Includes audio normalization, trimming, and standardized duration handling
- **High Performance**: Achieves strong classification accuracy through optimized neural network architecture

## 🏗️ Model Architecture

The system uses a **Compact Dense Neural Network** with the following architecture:
- **Input Layer**: 105 audio features
- **Hidden Layers**: 
  - Dense layer (512 neurons) + BatchNormalization + Dropout (0.3)
  - Dense layer (256 neurons) + BatchNormalization + Dropout (0.3)
  - Dense layer (128 neurons) + BatchNormalization + Dropout (0.2)
  - Dense layer (64 neurons) + Dropout (0.2)
- **Output Layer**: 7 neurons with softmax activation
- **Total Parameters**: 230,791 (901.53 KB)

## 🔬 Feature Engineering

The model extracts 105 comprehensive audio features:

### MFCC Features (39 features)
- **13 MFCC coefficients** (mean values)
- **13 MFCC coefficients** (standard deviation)
- **13 MFCC delta features** (mean values)

### Mel Spectrogram Features (48 features)
- **48 Mel-scale frequency bins** processed to dB scale

### Spectral Features (6 features)
- Spectral centroids (mean & std)
- Spectral rolloff (mean & std)
- Zero crossing rate (mean & std)

### Chroma Features (12 features)
- **12 chroma features** representing pitch class profiles

## 🗃️ Dataset

- **Dataset**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Audio Files**: 1,772 original files
- **Training Split**: 1,417 files (with augmentation: 7,085 samples)
- **Test Split**: 355 files
- **Sample Rate**: 16,000 Hz
- **Duration**: 3.5 seconds (standardized)

## 🛠️ Tech Stack

### Core Libraries
- **Deep Learning**: TensorFlow/Keras
- **Audio Processing**: Librosa
- **Data Science**: NumPy, Pandas, Scikit-learn
- **Web Interface**: Streamlit
- **Visualization**: Matplotlib, Seaborn

### Key Dependencies
```
streamlit>=1.25.0
tensorflow>=2.13.0
librosa>=0.10.0
scikit-learn>=1.3.0
numpy>=1.24.0
joblib>=1.3.0
```

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### 2. Create Virtual Environment
```bash
python -m venv emotion_recognition_env
source emotion_recognition_env/bin/activate  # On Windows: emotion_recognition_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Models
Ensure you have the following files in your project directory:
- `best_model_corrected.h5` (trained model)
- `scaler_.joblib` (feature scaler)

## 💻 Usage

### Web Application
Launch the interactive Streamlit web interface:
```bash
streamlit run app.py
```

Then:
1. Open your browser to `http://localhost:8501`
2. Upload an audio file (WAV, MP3, FLAC, M4A)
3. View the predicted emotion with confidence score

### Training Your Own Model
Use the provided Jupyter notebook:
```bash
jupyter notebook train-and-model.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Feature extraction
- Model training with callbacks
- Performance evaluation

## 📊 Model Performance

- **Architecture**: Dense Neural Network
- **Training Data**: 7,085 augmented samples
- **Test Data**: 355 samples
- **Features**: 105 audio features per sample
- **Classes**: 7 emotion categories
- **Optimization**: Adam optimizer with learning rate scheduling

## 📁 Project Structure

```
speech-emotion-recognition/
├── app.py                    # Streamlit web application
├── train-and-model.ipynb     # Training notebook
├── best_model_corrected.h5   # Trained model weights
├── scaler_.joblib           # Feature scaler
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## 🎬 Demo

The Streamlit app provides an intuitive interface where users can:
- Upload audio files in multiple formats
- See real-time emotion predictions
- View confidence scores with visual indicators
- Get color-coded emotion results with emojis

## 🔮 Future Enhancements

- **Real-time Audio Processing**: Live microphone input for instant emotion detection
- **Multi-language Support**: Extend to non-English speech patterns
- **Advanced Models**: Experiment with Transformer architectures and attention mechanisms
- **Mobile Deployment**: Create mobile app version using TensorFlow Lite
- **API Development**: REST API for integration with other applications
- **Emotion Intensity**: Add prediction of emotion intensity levels

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.



## 🙏 Acknowledgments

- **RAVDESS Dataset**: Thanks to the creators of the RAVDESS dataset
- **Librosa Library**: For excellent audio processing capabilities
- **TensorFlow Team**: For the deep learning framework
- **Streamlit**: For the intuitive web app framework

---

**Built with ❤️ for advancing emotion AI technology**
