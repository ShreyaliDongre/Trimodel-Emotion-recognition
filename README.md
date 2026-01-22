# Trimodal Emotion Recognition System
*(Facial + Speech + Text Analysis)*

## 📌 Project Overview
This project implements a **trimodal emotion recognition system** that combines **facial expressions**, **speech signals**, and **textual content** to predict human emotional states more accurately than unimodal systems.

Human emotions are inherently **multimodal**—facial cues, tone of voice, and spoken language often carry complementary emotional information. This system captures and fuses signals from all three modalities to produce a **robust and context-aware emotion prediction**.

---

## 🎯 Problem Statement
Traditional emotion recognition systems rely on a **single modality**, which leads to:
- Ambiguous predictions
- Sensitivity to noise or missing data
- Poor generalization in real-world settings

This project addresses these limitations by:
- Independently modeling each modality
- Learning modality-specific emotional features
- Combining predictions through **fusion strategies**

---

## 🧠 System Architecture
Facial Images ──► CNN ─┐
├─► Fusion ─► Final Emotion
Speech Audio ──► DNN ─┤
│
Text Input ──► NLP ─┘

Each modality is trained **independently**, and predictions are combined at the decision level.

---

## 🛠️ Technology Stack
- Python
- TensorFlow / Keras
- OpenCV
- Librosa
- NumPy, Pandas
- NLTK / VADER
- Pre-trained CNNs (EfficientNet, MobileNetV2, ResNet-18)

---

## 📂 Modalities & Implementation

---

## 1️⃣ Facial Emotion Recognition

### Dataset
- AffectNet
- 8 emotion classes:
  - Happy
  - Sad
  - Angry
  - Fear
  - Disgust
  - Surprise
  - Neutral
  - Contempt

### Preprocessing
- Face detection using Haar Cascade
- Image resizing and normalization
- Data augmentation

### Model
- Transfer learning using:
  - EfficientNet
  - MobileNetV2
  - ResNet-18 (experimentation)
- CNN-based classification

### Output
- Discrete emotion label  
- (Optional) Valence–Arousal estimation

---

## 2️⃣ Speech Emotion Recognition

### Dataset
- RAVDESS

### Feature Extraction
- MFCCs
- Chroma features
- Spectral Contrast

### Model
- Feedforward Deep Neural Network (Keras)

### Output
- Speech-based emotion prediction

---

## 3️⃣ Text-Based Emotion & Sentiment Analysis

### Dataset
- GoEmotions

### Techniques
- Emotion classification from text
- Sentiment scoring using **VADER**

### Purpose
- Capture semantic and contextual emotional cues
- Complement facial and speech signals

### Output
- Emotion category
- Sentiment polarity (positive / negative / neutral)

---

## 🔗 Fusion Strategy (Trimodal Integration)

### Fusion Type
- Late Fusion

### Approach
- Independent predictions from each modality
- Weighted averaging / majority voting
- Final emotion decision based on combined confidence

### Why Late Fusion?
- Modalities are heterogeneous (image, audio, text)
- Allows independent optimization
- Robust to missing modalities

---

## 📊 Evaluation Strategy

Each modality is evaluated independently using:
- Accuracy
- Confusion Matrix
- Precision, Recall, F1-score

Fusion performance is evaluated by:
- Comparing unimodal vs trimodal accuracy
- Observing robustness under noisy or partial input

---

## 🧩 Analytical Questions Addressed

- How does multimodal fusion improve emotion recognition accuracy?
- Which modality contributes most to specific emotions?
- How does text sentiment influence final emotion prediction?
- Can emotion be reliably inferred when one modality is missing or noisy?

---

## 🚧 Project Status
This project is a **research-oriented implementation** demonstrating:
- Multimodal learning
- Emotion-aware AI systems
- Practical fusion strategies

Suitable for:
- Academic evaluation
- Research paper development
- Advanced ML portfolios

---

## 🚀 Future Enhancements
- Attention-based fusion
- Temporal modeling using LSTMs or Transformers
- Real-time multimodal emotion recognition
- Multitask learning (emotion, sentiment, arousal)

---

## 📘 Key Learning Outcomes
- Multimodal deep learning
- Signal processing for emotion recognition
- Transfer learning
- NLP-based sentiment analysis
- Model fusion strategies

---

## 📄 License
Educational and research use only.


