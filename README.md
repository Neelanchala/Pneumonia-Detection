# 🩺 AI Pneumonia Detection System

An AI-assisted web application that analyzes chest X-ray images using a Convolutional Neural Network (CNN) to classify them as **Normal** or **Pneumonia**.

The system also generates **Grad-CAM visualizations**, produces downloadable **PDF diagnostic reports**, and maintains a **patient history database** with dashboard statistics.

> ⚠️ **Disclaimer:** This project is intended for educational and research purposes. It is not a medical diagnostic device and should not replace evaluation by a qualified healthcare professional.

---

## 📌 Project Overview

Pneumonia is a respiratory infection that can produce abnormalities visible in chest X-ray images.

This project combines:

- Deep Learning
- Computer Vision
- Explainable AI
- Flask Web Development
- SQLite Database
- Automated PDF Report Generation

The user uploads a chest X-ray along with basic patient information. The trained CNN analyzes the image and provides:

1. Predicted class
2. Prediction confidence
3. Grad-CAM heatmap
4. Attention overlay
5. AI-assisted interpretation
6. Clinical recommendation
7. Downloadable PDF report
8. Saved patient history

---

## 🚀 Features

### 🧠 CNN-Based Prediction

The trained CNN classifies chest X-ray images into:

- **Normal**
- **Pneumonia**

The system also displays the model's confidence score.

### 🔥 Grad-CAM Visualization

Grad-CAM is used to visualize image regions that contributed to the model's prediction.

The application generates:

- Original X-ray
- Grad-CAM heatmap
- AI attention overlay

This provides an additional interpretability layer instead of displaying only a classification result.

### 📄 Automated PDF Reports

After every analysis, the system generates a PDF report containing:

- Patient information
- Prediction
- Confidence
- Date and time
- AI interpretation
- Recommendation
- X-ray visualization
- Grad-CAM visualization
- Disclaimer

Each report is saved with a unique timestamped filename.

### 🗂️ Patient History

The application stores previous analyses using SQLite.

Users can:

- View previous reports
- Search by patient name
- Search by patient ID
- Open individual reports
- Delete reports

### 📊 Dashboard

The dashboard provides:

- Total reports
- Normal cases
- Pneumonia cases
- Average confidence
- Prediction distribution chart
- Recent reports

---

## 🏗️ System Architecture

```text
                 ┌─────────────────────┐
                 │        User         │
                 │ Upload X-ray        │
                 │ + Patient Details   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Flask Server     │
                 │       app.py        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      CNN Model      │
                 │     predict.py      │
                 └──────────┬──────────┘
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
          ┌──────────────┐    ┌──────────────┐
          │ Prediction   │    │   Grad-CAM   │
          │ Normal /     │    │ Visualization│
          │ Pneumonia    │    └──────┬───────┘
          └──────┬───────┘           │
                 │                   │
                 └─────────┬─────────┘
                           ▼
                 ┌─────────────────────┐
                 │    PDF Report       │
                 │ report_generator.py │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   SQLite Database   │
                 │     patients.db     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ History / Dashboard │
                 └─────────────────────┘

🧰 Technologies Used
Technology	         Purpose
Python	             Core programming language
TensorFlow / Keras	 CNN model
OpenCV	             Image processing and Grad-CAM
Flask	             Web application backend
HTML	             Frontend structure
CSS	                 User interface
JavaScript	         Frontend functionality
SQLite	             Patient/report database
ReportLab	         PDF report generation
Chart.js	         Dashboard visualization



🧠 CNN Model

The project uses a CNN trained for binary classification of chest X-ray images.

A simplified architecture is

Input X-ray
     │
     ▼
Rescaling
     │
     ▼
Convolution Block
     │
Batch Normalization
     │
ReLU
     │
Max Pooling
     │
     ▼
Convolution Block
     │
Batch Normalization
     │
ReLU
     │
Max Pooling
     │
     ▼
Flatten
     │
     ▼
Dense Layer
     │
Batch Normalization
     │
ReLU
     │
Dropout
     │
     ▼
Softmax
     │
     ├── Normal
     │
     └── Pneumonia


📁 Project Structure

Pneumonia-Detection/
│
├── app.py
├── database.py
├── predict.py
├── gradcam.py
├── report_generator.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   ├── index.html
│   ├── result.html
│   ├── history.html
│   ├── report_details.html
│   └── dashboard.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── uploads/
│   ├── heatmaps/
│   └── overlays/
│
└── reports/


🖥️ Application Workflow
Step 1 — Enter Patient Information

The user enters:

Patient name
Patient ID
Age
Gender
Doctor name
Hospital/clinic
Step 2 — Upload X-ray

Supported image formats:

.jpg
.jpeg
.png
Step 3 — AI Analysis

The CNN processes the image and produces:
Prediction
Confidence

Step 4 — Explainability

Grad-CAM generates:

Heatmap
Attention Overlay

Step 5 — Report Generation

A PDF report is automatically generated.

Step 6 — Database Storage

The analysis is saved in SQLite.

Step 7 — History and Dashboard

Previous analyses can be searched and viewed through:

/history

The dashboard is available at:

/dashboard

📊 Database

The application uses SQLite to store report information.

The database contains information such as:

Patient Name
Patient ID
Age
Gender
Doctor
Hospital
Prediction
Confidence
Original Image
Heatmap
Overlay
Report Path
Creation Date

The database file is excluded from GitHub because it contains locally generated application data.

🛡️ Data and Privacy

This project is designed as a local educational application.

The repository does not include:

Patient database
Uploaded X-rays
Generated reports
Generated heatmaps
Generated overlays

These files are excluded using .gitignore.

Do not upload real patient information or medical images to a public repository.

⚠️ Limitations
Model Limitations

Prediction quality depends on:

Training dataset
Dataset size
Class balance
Image quality
Preprocessing
Model architecture
Training procedure

A high confidence score does not necessarily mean the prediction is medically correct.

Grad-CAM Limitations

Grad-CAM provides an approximate visualization of model attention. It should not be interpreted as definitive medical localization of pneumonia.

Clinical Limitations

This system is not a replacement for a radiologist or physician.

The output should only be considered an AI-assisted prediction for educational and research purposes.

🔮 Future Improvements

Possible future improvements include:

Transfer learning using MobileNet, EfficientNet, or ResNet
Better dataset balancing
Improved validation methodology
ROC-AUC evaluation
Precision, recall and F1-score analysis
Better calibration of confidence scores
Multi-class chest disease classification
DICOM image support
Authentication and user accounts
Cloud deployment
Secure medical data storage
Model version management
Automated model performance monitoring
🎯 Learning Objectives

This project demonstrates practical experience with:

Convolutional Neural Networks
Image classification
Medical image processing
Explainable AI
Grad-CAM
TensorFlow/Keras
Flask
SQLite database integration
PDF generation
Frontend development
Git/GitHub
👨‍💻 Author

Neelanchala Nayak

B.Tech — Electronics & Communication Engineering

GitHub:

https://github.com/Neelanchala

📜 Disclaimer

This project is developed for educational and research purposes.

It is not a certified medical device, and its predictions should not be used for diagnosis, treatment, or other medical decisions.

Always consult a qualified healthcare professional for medical evaluation.


After pasting and saving:

```bash
git add README.md
git commit -m "Improve project documentation"
git push



