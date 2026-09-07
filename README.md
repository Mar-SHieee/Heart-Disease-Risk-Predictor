# ❤️ Heart Disease Risk Predictor

This project is a complete end-to-end machine learning workflow for predicting heart disease risk. Rather than simply loading a dataset and training a model, I started with a thorough statistical investigation to understand the data and identify the factors that distinguish patients with heart disease from healthy individuals. I then built three custom, leakage-free feature engineering transformers from scratch, ensuring that every learned statistic was calculated only from the training data within each cross-validation fold. After that, I trained, cross-validated, and tuned four tree-based models, compared their performance on unseen data, and selected the best-performing model for deployment. Finally, I used SHAP to interpret the model's predictions and understand which features were driving its decisions. The entire pipeline was then packaged into a working Streamlit application.

The project uses the Cleveland Heart Disease dataset from UCI, available on Kaggle as [cherngs/heart-disease-cleveland-uci](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci).

## 📂 Project Structure

Heart Disease Risk Predictor/
├── app.py
├── preprocessing.py
├── heart_disease_pipeline.joblib
├── requirements.txt
├── Heart Disease Risk Predictor.ipynb
└── README.md

## File Overview

| File | Purpose |
|---|---|
| `app.py` | Streamlit application used for deployment |
| `preprocessing.py` | Custom scikit-learn transformers used inside the pipeline |
| `heart_disease_pipeline.joblib` | Final trained and serialized machine learning pipeline |
| `requirements.txt` | Project dependencies |
| `Heart Disease Risk Predictor.ipynb` | Complete analysis, experimentation, and modeling workflow |
| `README.md` | Project documentation |

> **⚠️ Important:** `app.py` loads `heart_disease_pipeline.joblib` using `joblib.load()`. The saved pipeline contains three custom transformer classes defined in `preprocessing.py`. Because serialized scikit-learn pipelines need access to the original class definitions when they are loaded, `preprocessing.py` must remain available alongside `app.py`. Changing or removing those classes can prevent the saved pipeline from loading correctly.

## 🧠 Project Workflow

### Exploratory Data Analysis and Statistical Analysis
Before training any model, I focused on understanding the dataset and the relationships between its features and the target.
I first examined the class distribution and performed univariate analysis on individual features such as:
*   `chol`
*   `thalach`
*   `cp`
*   `exang`
*   `sex`

I then performed bivariate statistical analysis:
*   **T-tests** for numeric features
*   **Chi-square tests** for categorical features
*   **Correlation analysis** to identify features most strongly associated with heart disease

I also used PCA and LDA to visualize the separability between the two classes. These techniques were used for exploration and visualization, not as the final dimensionality reduction or modeling approach.

### Custom Leakage-Free Feature Engineering
Feature engineering was designed with a strong focus on preventing data leakage. I built three custom scikit-learn-compatible transformers:

*   **`RowWiseFeatureAdder`**: Creates features directly from each individual row without learning any statistics from the dataset. Examples include `bp_chol_ratio`, `age_hr_interaction`, and `chest_pain_ecg`. Because these transformations are purely row-wise, they do not require fitting.
*   **`RiskScoreAdder`**: Combines the strongest risk-related features into a single composite `risk_score`. The score uses feature weights based on their correlation with the target and applies min-max scaling. Most importantly, the correlations and scaling parameters are learned *only* from the training portion of each cross-validation fold.
*   **`OutlierCapper`**: Instead of removing observations containing outliers, this transformer caps extreme values using IQR-based bounds. The lower and upper bounds are calculated independently for each training fold, ensuring that information from the validation fold never influences preprocessing.

**Why This Matters**
All learned statistics (including correlations, scaling parameters, quantile, and IQR bounds) are fitted only on the corresponding training data. This prevents information from the validation or test sets from influencing the feature engineering process and keeps the evaluation reliable.

## 🌳 Model Training and Selection

I followed a structured modeling process rather than jumping directly into hyperparameter tuning.

**Baseline Models**
I first trained four tree-based classification models using their baseline configurations:
*   Decision Tree
*   Random Forest
*   Gradient Boosting
*   XGBoost

The baseline models established a clear performance benchmark and helped identify how each algorithm performed before optimization.

**Cross-Validation and Model Comparison**
I evaluated the baseline models using 5-fold Stratified Cross-Validation, preserving the class distribution across folds. This allowed me to compare the models more reliably and assess how consistently they performed across different subsets of the training data.

**Hyperparameter Tuning**
After establishing the baseline results, I optimized each model using `RandomizedSearchCV`. The goal was to determine whether tuning the models could meaningfully improve their performance beyond the baseline configurations.

**Final Model Selection**
I compared the tuned models based on their validation performance and then evaluated the selected model on a completely held-out test set. Random Forest achieved the strongest overall performance, reaching an AUC-ROC of approximately 0.93 on the held-out test set. It was therefore selected as the final model and deployed in the Streamlit application.

## 🔍 Model Interpretability

Model performance alone is not enough for a project like this. I also wanted to understand why the models were making their predictions. I performed several interpretability analyses:

*   **Feature Importance Comparison:** I compared feature importance across all four models. Interestingly, `risk_score` ranked as the most important feature across every model.
*   **Decision Tree Analysis:** I examined the tuned Decision Tree's actual splits to understand the decision-making structure and identify which features were used at different stages of classification.
*   **XGBoost Learning Curves:** I analyzed XGBoost's learning curves to investigate its training behavior and evaluate the bias-variance tradeoff.
*   **SHAP Analysis:** I used SHAP (SHapley Additive exPlanations) to investigate both global and individual model behavior. This included SHAP summary plots to understand global feature impact, and individual force plots to explain specific predictions. This provided a more detailed view of how individual features contributed to the model's output.

**Additional Experiments**
I also tested two alternative approaches:
*   Cost-sensitive learning using `scale_pos_weight`
*   Stacking the four models together

Cost-sensitive learning produced only a minor improvement because the dataset's class imbalance was relatively mild. Stacking also failed to outperform the strongest individual models, so I ultimately kept the standalone Random Forest model.

## 🚀 Deployment

After selecting the final model, I saved the complete preprocessing and prediction workflow as a single serialized pipeline:

joblib.dump(pipeline, "heart_disease_pipeline.joblib")

The saved pipeline contains the preprocessing steps, custom feature engineering, and trained Random Forest model. The Streamlit application loads this pipeline directly, meaning no retraining is required when the application starts.

The deployed application allows users to enter patient information and receive:
*   A predicted heart disease class
*   The corresponding prediction probability
*   A clear risk verdict

## 💻 How to Run the Application

**1. Create a Virtual Environment**
python -m venv venv

**2. Activate the Environment**
*On Windows:*
venv\Scripts\activate

*On macOS/Linux:*
source venv/bin/activate

**3. Install Dependencies**
pip install -r requirements.txt

**4. Start the Streamlit App**
streamlit run app.py

The application will be available at: `http://localhost:8501`. Enter the patient's values in the form and click Predict to generate the model's prediction and probability.

## 📋 Input Features

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

## 🧩 Key Highlights

*   Complete EDA and statistical analysis before modeling
*   T-tests and chi-square tests for statistical feature analysis
*   PCA and LDA for class separability visualization
*   Three custom scikit-learn transformers
*   Explicit data leakage prevention throughout preprocessing and cross-validation
*   Feature engineering with a custom composite `risk_score`
*   IQR-based outlier capping instead of deleting observations
*   Comparison of four tree-based classification models
*   5-fold Stratified Cross-Validation
*   Hyperparameter tuning with `RandomizedSearchCV`
*   Final Random Forest model with approximately 0.93 AUC-ROC on the held-out test set
*   Cross-model feature importance analysis
*   SHAP-based model interpretability
*   Bias-variance analysis using learning curves
*   Cost-sensitive learning experiment
*   Stacking experiment
*   End-to-end serialized ML pipeline
*   Interactive Streamlit deployment