# 🎭 Real-Time Emotion Recognition with DeepFace & OpenCV

This project uses AI to detect human emotions from facial expressions in real-time through your webcam

Built with:
- [DeepFace](https://github.com/serengil/deepface) for facial emotion analysis
- OpenCV for webcam access and frame processing

---

## 🛠️ Features

- Live webcam feed with facial detection
- Emotion prediction: Happy, Sad, Angry, Surprise, Neutral, Fear, Disgust
- Confidence bars for all emotions
- Uses **pre-trained deep learning models**
- Live timestamp and FPS counter
- Lightweight and easy to run locally

---

## 📦 Dependencies

- Python 3.7+
- OpenCV (`opencv-python`)
- DeepFace
- TensorFlow (2.x)
- tf-keras (for TensorFlow 2.19+)

---

## ⚠️ Limitations

- Accuracy depends on lighting and camera quality
- May switch emotions rapidly without smoothing
- Designed for one face at a time (no multi-face tracking)

---