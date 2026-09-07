# ❤️ Heart Disease Risk Predictor

This isn't a quick "load a dataset, call `.fit()`, done" notebook. I ran a full statistical investigation before touching a single model — t-tests, chi-square tests, PCA, LDA — to actually understand what separates a heart disease patient from a healthy one. Then I hand-built three custom leakage-free transformers from scratch to engineer features properly instead of leaning on shortcuts that quietly leak information across cross-validation folds. I trained, cross-validated, and tuned four different tree-based models, picked a winner based on real held-out performance, then went back in with SHAP to explain exactly *why* it predicts what it predicts. And I didn't stop at a notebook — I shipped it as a working Streamlit app.

I used the **Cleveland Heart Disease dataset** (UCI / [`cherngs/heart-disease-cleveland-uci`](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) on Kaggle) for this.

---

## 📂 What's in this folder

```
Heart Disease Risk Predictor/
├── app.py                              # The Streamlit app I deployed
├── preprocessing.py                    # My custom sklearn transformers, used inside the pipeline
├── heart_disease_pipeline.joblib       # My final trained pipeline, serialized
├── requirements.txt                    # Pinned dependencies
├── Heart_Disease_Risk_Predictor.ipynb  # The full notebook — all my analysis and experiments
└── README.md
```

> ⚠️ One thing I learned the hard way: `app.py` loads `heart_disease_pipeline.joblib` with `joblib.load`, and that pipeline contains three custom transformer classes I wrote myself. Pickle needs those exact classes importable at load time, which is why `preprocessing.py` has to sit right next to `app.py` — if it's missing or the class code changes, loading breaks.

---

## 🧠 How I actually got here

### 1. I started with EDA
Before touching any model, I wanted to actually understand the data. I checked the class balance, then looked at each feature individually — things like `chol`, `thalach`, `cp`, `exang`, `sex`. Then I went further with bivariate analysis: I ran **t-tests** on the numeric features and **chi-square tests** on the categorical ones against the target, and ranked the features most correlated with heart disease. I also ran **PCA** and **LDA**, purely to visualize how separable the two classes were — not for actual dimensionality reduction.

### 2. I built my own leakage-free feature engineering
This was the part I was most careful about. I built three custom `sklearn`-compatible transformers, and I made sure every single statistic they learn — correlations, scalers, quantile bounds — gets fit **only** on that fold's training data, so nothing leaks across cross-validation folds:
- **`RowWiseFeatureAdder`** — pure row-wise features (`bp_chol_ratio`, `age_hr_interaction`, `chest_pain_ecg`) that don't need fitting at all
- **`RiskScoreAdder`** — I combined the strongest risk features into one composite `risk_score`, weighted by their correlation with the target and min-max scaled
- **`OutlierCapper`** — instead of dropping outlier rows, I capped them using IQR bounds, computed independently per fold

### 3. I trained and tuned four different models
I trained a Decision Tree, Random Forest, Gradient Boosting, and XGBoost, cross-validated each with 5-fold stratified CV, compared them, then tuned all four with `RandomizedSearchCV`.

**Random Forest ended up winning** after tuning, with an AUC-ROC of about 0.93 on the held-out test set — that's the one I deployed.

### 4. I dug into why the models made the decisions they did
I didn't want a black box, so I went further:
- Compared feature importances across all four models — `risk_score` came out on top for every single one
- Looked at the tuned Decision Tree's actual splits to understand its logic
- Plotted XGBoost's learning curves and analyzed bias-variance tradeoffs
- Ran **SHAP** (summary plots and individual force plots) to explain specific predictions
- Tested cost-sensitive learning (`scale_pos_weight`) — it barely moved the needle, since the class imbalance turned out to be mild
- Tried stacking the models together — it didn't beat XGBoost or Random Forest alone, so I dropped it

### 5. I packaged it for deployment
Once I picked the tuned Random Forest, I saved the whole pipeline — preprocessing steps and model together — with `joblib.dump()` as `heart_disease_pipeline.joblib`. That's the exact file `app.py` loads, so there's no retraining needed to make predictions.

---

## 🚀 How to run the app

**1. Install dependencies** (I'd recommend a virtual environment):
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

**2. Run it:**
```bash
streamlit run app.py
```

It'll open automatically in your browser at `http://localhost:8501`. Fill in the patient's values in the form and hit **Predict** to get a risk verdict and probability.

---

## 📋 The input features

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment (0–2) |
| `ca` | Number of major vessels colored by fluoroscopy (0–3) |
| `thal` | Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect) |

## ⚠️ Disclaimer

I built this as a learning/portfolio project. It's **not a medical diagnosis tool** — please don't use it as one. Always consult a real healthcare professional.
