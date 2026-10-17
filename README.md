# Credit Risk Classification Using Logistic Regression

> A full-stack machine learning web application that classifies loan applicants into **Low**, **Medium**, and **High** risk categories using Logistic Regression — trained on the UCI German Credit dataset.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [ML Methodology](#ml-methodology)
- [Technologies Used](#technologies-used)
- [Folder Structure](#folder-structure)
- [Installation & Setup](#installation--setup)
- [Model Training](#model-training)
- [Running the Application](#running-the-application)
- [Application Routes](#application-routes)
- [Evaluation Results](#evaluation-results)
- [Risk Classification Strategy](#risk-classification-strategy)
- [Deployment](#deployment)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Project Overview

This project implements an **end-to-end credit risk assessment system**. Given 20 financial and personal attributes of a loan applicant (sourced from the UCI German Credit dataset), the system:

1. **Preprocesses** the data using `StandardScaler` (numerical) and `OneHotEncoder` (categorical) inside a `ColumnTransformer` pipeline.
2. **Trains** a `LogisticRegression` classifier (with `class_weight='balanced'` to handle class imbalance).
3. **Classifies** applicants into three risk tiers based on predicted default probability.
4. **Serves** predictions through a user-friendly Flask web interface with history tracking.

---

## Objectives

- Build an interpretable credit risk scoring model using Logistic Regression.
- Deploy a real-time prediction API via Flask.
- Provide an interactive UI for loan officers / analysts.
- Track prediction history per browser session using SQLite.
- Include an AI credit advisor chat feature.

---

## ML Methodology

### Dataset

| Property       | Detail                                 |
|----------------|----------------------------------------|
| **Source**     | [UCI ML Repository – German Credit](https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)) |
| **Samples**    | 1,000 applicants                       |
| **Features**   | 20 (13 categorical + 7 numerical)      |
| **Target**     | Binary: `1 = Good`, `2 = Bad`          |
| **Class Ratio**| ~70% Good, ~30% Bad                   |

### Preprocessing Pipeline

```
Raw Input
   │
   ├── Numerical Features (7)
   │       └── StandardScaler (zero mean, unit variance)
   │
   └── Categorical Features (13)
           └── OneHotEncoder (handle_unknown='ignore')
           
   └── ColumnTransformer → Logistic Regression (max_iter=1000, class_weight='balanced')
```

### Risk Tier Mapping

After predicting the probability of default (`P(Bad)`):

| Probability Range | Risk Category | Action Recommended     |
|-------------------|---------------|------------------------|
| `P ≤ 0.30`        | **Low Risk**  | Approve loan           |
| `0.30 < P ≤ 0.60` | **Medium Risk** | Review manually      |
| `P > 0.60`        | **High Risk** | Decline / extra review |

### Model Selection Rationale

Logistic Regression was selected because:
- **Interpretability**: Coefficients directly show feature impact.
- **Speed**: Fast training and inference for real-time web use.
- **Baseline**: Strong baseline for credit scoring before exploring tree-based models.
- **Regulatory fit**: Financial institutions often require explainable AI decisions.

---

## Technologies Used

| Layer        | Technology                         |
|--------------|-------------------------------------|
| Language     | Python 3.10+                        |
| ML Framework | scikit-learn 1.3+                   |
| Data         | pandas, numpy                       |
| Web Backend  | Flask 3.1                           |
| Templating   | Jinja2                              |
| Frontend     | HTML5, CSS3, JavaScript             |
| Storage      | SQLite (prediction history)         |
| Serialization| joblib                              |
| Testing      | pytest                              |
| Deployment   | Docker, Render (render.yaml)        |

---

## Folder Structure

```
Credit Risk Classification/
├── app.py                  # Flask application entry point
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container definition
├── render.yaml             # Render deployment config
├── .gitignore              # Git ignore rules
│
├── src/                    # Core Python modules
│   ├── __init__.py
│   ├── config.py           # Paths, hyperparameters, feature lists
│   ├── preprocessing.py    # Dataset download, loading, cleaning
│   ├── model_utils.py      # Training, prediction, metrics
│   └── ai_chat.py          # AI Credit Advisor chat engine
│
├── models/                 # Trained model artifacts (auto-generated)
│   ├── logistic_regression_model.joblib
│   ├── preprocessor.joblib
│   └── metrics.joblib
│
├── data/                   # Dataset directory (auto-downloaded)
│   └── README.md
│
├── templates/              # Jinja2 HTML templates
│   ├── base.html           # Shared layout
│   ├── index.html          # Home page
│   ├── predict.html        # Prediction form
│   ├── result.html         # Prediction results
│   ├── dashboard.html      # Model metrics dashboard
│   ├── history.html        # Prediction history
│   ├── about.html          # Project info
│   ├── chat.html           # AI advisor chat
│   ├── 404.html            # Error pages
│   └── 500.html
│
├── static/                 # Frontend assets
│   ├── css/style.css
│   └── js/script.js
│
├── tests/                  # Automated tests (pytest)
│   ├── __init__.py
│   ├── test_app.py
│   └── test_model.py
│
└── screenshots/            # UI screenshots (optional)
```

---

## Installation & Setup

### Prerequisites

- Python **3.10** or higher
- `pip` package manager
- Internet connection (to download the dataset)

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/Credit-Risk-Classification.git
cd Credit-Risk-Classification
```

### Step 2: Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Model Training

The dataset is automatically downloaded from the UCI ML Repository during training.

```bash
python train_model.py
```

This will:
1. Download the German Credit dataset from UCI
2. Load and clean the data
3. Train the Logistic Regression model
4. Evaluate on a 20% test split
5. Save artifacts to the `models/` directory:
   - `logistic_regression_model.joblib`
   - `preprocessor.joblib`
   - `metrics.joblib`

> **Note:** Pre-trained model artifacts are already included in the repository. Re-training is optional.

---

## Running the Application

```bash
python app.py
```

Then open your browser at: **http://localhost:5000**

### Using Docker

```bash
docker build -t credit-risk-app .
docker run -p 5000:5000 credit-risk-app
```

---

## Application Routes

| Route         | Method    | Description                           |
|---------------|-----------|---------------------------------------|
| `/`           | GET       | Home page                             |
| `/predict`    | GET, POST | Credit risk prediction form           |
| `/result`     | GET       | Prediction result display             |
| `/dashboard`  | GET       | Model performance metrics             |
| `/history`    | GET       | Per-browser prediction history        |
| `/about`      | GET       | Project information                   |
| `/chat`       | GET       | AI Credit Advisor chat interface      |
| `/api/chat`   | POST      | AI chat JSON API                      |

---

## Evaluation Results

> Metrics generated on a 20% held-out test split (200 samples, random_state=42):

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~77%   |
| Precision | ~76%   |
| Recall    | ~77%   |
| F1 Score  | ~76%   |
| ROC AUC   | ~80%   |

> Exact values are stored in `models/metrics.joblib` and displayed on the `/dashboard` page.

---

## Risk Classification Strategy

The model predicts the **probability of default** (Bad credit) using `predict_proba()`. This continuous probability is then thresholded into three risk categories:

```python
if P(Bad) <= 0.30:
    risk = "Low Risk"      # Approve
elif P(Bad) <= 0.60:
    risk = "Medium Risk"   # Manual review
else:
    risk = "High Risk"     # Decline
```

This approach is more useful than a hard binary classification because it allows loan officers to calibrate their decision thresholds based on business risk appetite.

---

## Deployment

The project includes a [`render.yaml`](render.yaml) and [`Dockerfile`](Dockerfile) for one-click deployment to [Render](https://render.com).

### Environment Variables for Production

| Variable     | Description                   | Default         |
|--------------|-------------------------------|-----------------|
| `SECRET_KEY` | Flask session secret          | (change this!)  |
| `FLASK_DEBUG`| Enable debug mode             | `False`         |
| `GEMINI_API_KEY` | Optional Gemini AI key for chat | None        |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Future Enhancements

- [ ] Add XGBoost / Random Forest models for comparison
- [ ] SHAP explainability for individual predictions
- [ ] User authentication and per-user history
- [ ] REST API with OpenAPI / Swagger documentation
- [ ] Automated CI/CD with GitHub Actions
- [ ] Model retraining pipeline with MLflow tracking
- [ ] Deploy to AWS / GCP / Azure

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Author

**Venkata Sriram Kumar Mallisetty**  
Email: venkatasriramkumarmallisetty@gmail.com  
GitHub: [@venkatasriramkumarmallisetty-bit](https://github.com/venkatasriramkumarmallisetty-bit)

---

*Built with Python, Flask, scikit-learn | UCI German Credit Dataset*
