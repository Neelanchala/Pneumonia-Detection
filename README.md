# 🩺 AI Pneumonia Detection System

An AI-powered web application that detects pneumonia from Chest X-ray images using a Convolutional Neural Network (CNN). The system generates Grad-CAM visualizations, AI-assisted diagnostic reports, patient history, and an analytics dashboard.

---

## 📌 Features

- Chest X-ray upload
- CNN-based Pneumonia Detection
- Grad-CAM Heatmap Generation
- AI Attention Overlay
- Patient Information Management
- PDF Diagnostic Report Generation
- Patient History Database (SQLite)
- Search Patient Records
- Dashboard with Statistics
- Confidence Score Visualization
- Responsive Web Interface

---

## 🛠 Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript

### Backend
- Flask (Python)

### AI / Deep Learning
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

### Database
- SQLite

### PDF Generation
- ReportLab

---

## 📂 Project Structure

```
Pneumonia-Detection/
│
├── app.py
├── database.py
├── predict.py
├── gradcam.py
├── report_generator.py
├── requirements.txt
├── README.md
│
├── reports/
├── static/
│   ├── css/
│   ├── uploads/
│   ├── heatmaps/
│   ├── overlays/
│   └── images/
│
└── templates/
    ├── index.html
    ├── result.html
    ├── history.html
    ├── dashboard.html
    └── report_details.html
```

---

## ⚙ Installation

Clone the repository

```bash
git clone <repository-link>
```

Move into the project

```bash
cd Pneumonia-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Download the trained model (**my_cnn_best.keras**) and place it in the project root.

Run the application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

## 🚀 Workflow

1. Enter patient information.
2. Upload a Chest X-ray image.
3. AI predicts **Normal** or **Pneumonia**.
4. Grad-CAM heatmap is generated.
5. Attention overlay is created.
6. PDF report is generated.
7. Patient data is stored in SQLite.
8. Dashboard and history are updated.

---

## 📊 Dashboard

The dashboard provides:

- Total Reports
- Normal Cases
- Pneumonia Cases
- Average Confidence
- Pie Chart Visualization
- Recent Reports

---

## 📄 Generated Report

Each report contains:

- Patient Information
- AI Prediction
- Confidence Score
- AI Interpretation
- Clinical Recommendation
- Original X-ray
- Grad-CAM Heatmap
- AI Overlay
- Disclaimer

---

## 📸 Screenshots

Add screenshots here.

### Home Page

```
images/home.png
```

### Prediction Result

```
images/result.png
```

### Dashboard

```
images/dashboard.png
```

### Patient History

```
images/history.png
```

---

## 🔮 Future Improvements

- Better CNN architecture
- MobileNet/EfficientNet support
- DICOM image support
- Multi-class disease detection
- User Authentication
- Cloud Deployment
- REST API
- Doctor Portal

---

## ⚠ Disclaimer

This application is intended for educational and research purposes only. It should not replace professional medical diagnosis or clinical decision-making.

---

## 👨‍💻 Author

**Neelanchala Nayak**

B.Tech – Electronics & Communication Engineering

NIST University