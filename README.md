<div align="center">

```
███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ██████╗ ██╗███████╗██╗  ██╗
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔══██╗██║██╔════╝██║ ██╔╝
██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║██████╔╝██║███████╗█████╔╝ 
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══██╗██║╚════██║██╔═██╗
██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝██║  ██║██║███████║██║  ██╗
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝
```

### 🧠 AI-Powered Stroke Risk Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c?style=for-the-badge)](https://matplotlib.org)

> **Every second counts in stroke detection.**  
> NeuroRisk uses machine learning to identify high-risk patients before it's too late.

</div>

---

## 💡 The Idea

Stroke is the **2nd leading cause of death globally**, responsible for approximately 11% of all deaths (WHO). The tragedy? Up to **80% of strokes are preventable** with early detection and timely intervention.

The challenge in clinical settings is that stroke risk is determined by a complex interplay of factors — age, blood pressure, glucose levels, lifestyle habits — that are difficult to evaluate manually at scale.

**NeuroRisk** solves this by training multiple machine learning models on real patient data to predict stroke risk instantly. The system is deliberately tuned to prioritize **Recall over Precision** — in medicine, missing a real stroke (false negative) is far more dangerous than a false alarm (false positive).

> 🎯 **Goal:** Minimize false negatives. Flag every at-risk patient. Let doctors decide.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **4 ML Models** | Decision Tree, Random Forest, SVC, Gradient Boosting — trained & compared |
| 📊 **Analytics Dashboard** | ROC curves, feature importances, model comparison charts |
| ⚡ **Real-time Prediction** | Instant risk score with animated gauge & risk level |
| 🎯 **High Recall (~89%)** | Threshold tuned to catch maximum stroke cases |
| ⚖️ **Imbalance Handling** | `class_weight="balanced"` + threshold tuning for skewed data |
| 🔍 **Risk Factor Detection** | Identifies which specific factors are driving the risk |
| 🌐 **Full-Stack Web App** | Clean, dark-themed UI built with Flask + vanilla JS |

---

## 🗂️ Project Structure

```
stroke_app/
│
├── 📄 app.py                        # Flask app — routes & prediction logic
├── 📄 train_models.py               # ML pipeline — train, evaluate, save
├── 📄 requirements.txt              # Python dependencies
├── 📄 README.md
│
├── 📁 models/
│   ├── artifacts.pkl                # Saved model + encoders (auto-generated)
│   └── results.json                 # Performance metrics (auto-generated)
│
├── 📁 static/
│   └── img/                         # Charts generated after training
│       ├── model_comparison.png
│       ├── roc_curves.png
│       ├── feature_importance.png
│       └── class_dist.png
│
└── 📁 templates/
    ├── index.html                   # Prediction form UI
    └── analytics.html               # Model analytics dashboard
```

---

## 📊 Dataset

**Source:** [Kaggle — Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)  
**Records:** 5,110 real patient entries  
**Stroke Rate:** ~4.87% (heavily imbalanced — handled via balanced class weights)

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Patient age |
| `bmi` | Numerical | Body Mass Index |
| `avg_glucose_level` | Numerical | Average blood glucose (mg/dL) |
| `hypertension` | Binary | 0 = No, 1 = Yes |
| `heart_disease` | Binary | 0 = No, 1 = Yes |
| `gender` | Categorical | Male / Female |
| `ever_married` | Categorical | Yes / No |
| `work_type` | Categorical | Private / Self-employed / Govt / etc. |
| `Residence_type` | Categorical | Urban / Rural |
| `smoking_status` | Categorical | Never / Formerly / Currently / Unknown |
| `stroke` | **Target** | 0 = No Stroke, 1 = **Stroke** |

---

## 🤖 Models & Results

All models trained with `class_weight="balanced"` and prediction threshold set to **0.30** to maximize Recall.

```
┌─────────────────────┬──────────┬───────────┬──────────┬─────────┐
│ Model               │ Recall ↑ │ Precision │ F1 Score │ AUC-ROC │
├─────────────────────┼──────────┼───────────┼──────────┼─────────┤
│ Decision Tree       │  86.6%   │   37.6%   │  52.4%   │  0.750  │
│ Random Forest  ⭐   │  88.8%   │   37.1%   │  52.3%   │  0.792  │
│ SVC                 │  69.0%   │   50.4%   │  58.2%   │  0.785  │
│ Gradient Boosting   │  65.3%   │   51.4%   │  57.6%   │  0.780  │
└─────────────────────┴──────────┴───────────┴──────────┴─────────┘
```

> ⭐ **Random Forest** selected as the best model based on highest Recall (88.8%)

### Why Recall over Precision?

```
False Negative (missed stroke) → Patient sent home → Stroke occurs → Potentially fatal ❌
False Positive (wrong alarm)   → Extra tests ordered → No harm done ✅
```

In clinical ML, **missing a positive case costs more than a false alarm.** Recall is king.

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/stroke-prediction-webapp.git
cd stroke-prediction-webapp
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) and place `healthcare-dataset-stroke-data.csv` inside the `stroke_app/` folder.

### 4. Train the models
```bash
python train_models.py
```
This will:
- Preprocess the dataset
- Train all 4 ML models
- Save the best model to `models/artifacts.pkl`
- Generate performance charts in `static/img/`

### 5. Run the web app
```bash
python app.py
```

### 6. Open in browser
```
http://localhost:5001
```

---

## 🖥️ App Pages

### 🏠 Prediction Page (`/`)
- Input patient parameters via an interactive form
- Get instant stroke risk probability with animated gauge
- See risk level: Low / Moderate / High / Very High
- View identified risk factors

### 📈 Analytics Page (`/analytics`)
- Model comparison bar chart
- ROC curves for all 4 models
- Feature importance rankings
- Class distribution pie chart
- Detailed metrics table

---

## 🔬 ML Pipeline

```
Raw CSV Data
     │
     ▼
┌─────────────────────────┐
│  Data Preprocessing     │
│  • Drop ID column       │
│  • Fill missing BMI     │
│  • Label Encoding       │
│  • Train/Test Split     │
│  • Standard Scaling     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Model Training         │
│  • Decision Tree        │
│  • Random Forest        │
│  • SVC                  │
│  • Gradient Boosting    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Imbalance Handling     │
│  • class_weight=balance │
│  • Threshold = 0.30     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Evaluation             │
│  • Recall (priority)    │
│  • Precision, F1, AUC   │
│  • Best model saved     │
└─────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, Flask |
| **ML** | Scikit-learn |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib |
| **Frontend** | HTML, CSS, JavaScript |
| **Fonts** | Space Mono, DM Sans |

---

## ⚠️ Disclaimer

> This application is built for **research and educational purposes only.**  
> It is **not** a certified medical device and should **not** be used as a substitute for professional medical diagnosis or clinical decision-making.  
> Always consult a qualified healthcare professional.

---

<div align="center">

Made with ❤️ by Kanika

⭐ Star this repo if you found it useful!

</div>
