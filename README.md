# AI Transformer Health Monitor

This repository contains the Honors Contract project for **EECE 4201 – Energy Conversion**.

The goal is to develop an **AI-based method for monitoring the health condition of power transformers** using electrical and thermal parameters.

---

## 🧩 Overview
Transformers are key components in power networks, and their failure can lead to costly downtime.  
This project demonstrates how artificial intelligence can classify transformer operating states—such as **Normal**, **Overload**, **Overheat**, and **Fault**—based on electrical measurements.

---

## ⚙️ Features
- Synthetic data generation simulating transformer behavior
- Machine-learning classification using **Random Forest**
- Visual analytics: confusion matrix & feature importance
- Ready for **MATLAB/Simulink** integration (replace synthetic dataset with simulation data)

---

## 🧠 Requirements
Python 3.10+  
```
pip install numpy pandas scikit-learn matplotlib joblib
```

---

## ▶️ Usage
Run the training pipeline:
```
python train_ai_transformer_health.py
```

This will:
- Load the dataset (`transformer_synthetic_dataset.csv`)
- Train and evaluate the Random Forest classifier
- Save:
  - `confusion_matrix.png`
  - `feature_importance.png`
  - `model_random_forest.pkl`

---

## 📂 Project Structure
```
.
├── transformer_synthetic_dataset.csv
├── train_ai_transformer_health.py
├── model_random_forest.pkl
├── confusion_matrix.png
├── feature_importance.png
└── README.md
```

---

## 📊 Results
- Achieved >90% accuracy in distinguishing transformer health states
- Most important features: **RMS current**, **Temperature**, and **Losses**

---

## 🚀 Future Work
- Integrate real transformer data from MATLAB/Simulink
- Implement deep-learning models for waveform-based classification
- Develop real-time monitoring dashboard

---

## 🧾 Credits
Developed by **Brayden O’Briant**  
Supervised by **Dr. Hasan Ali**  
Department of Electrical & Computer Engineering  
University of Memphis, Fall 2025
