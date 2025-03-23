# 🏥 Insurance Charges Prediction using Gradient Boosting Regression

## 🔍 Overview

This project aims to **predict individual medical insurance charges** using a variety of regression techniques, with a particular focus on **Gradient Boosting**. By incorporating demographic and lifestyle data (age, sex, BMI, number of children, smoking status, and region), the model provides estimates to guide cost expectations and inform healthcare planning.

## ✨ Key Features

- 🤖 **Comprehensive Model Comparison**: Evaluates 12 different regression models (e.g., Linear, Lasso, XGBoost, etc.)  
- 🏗️ **Robust Data Preprocessing**: Combines numeric scaling, one-hot encoding, and pipeline-based transformations  
- 🔬 **In-depth Exploratory Data Analysis (EDA)**: Charts, plots, and statistical summaries to identify critical factors  
- 🔧 **Hyperparameter Tuning**: Reduces overfitting and optimizes Gradient Boosting Regressor performance

## 📦 Dataset

- **Source**: `insurance.csv`  
- **Attributes**:  
  - `age` — Age of the individual  
  - `sex` — Biological sex (`male`, `female`)  
  - `bmi` — Body Mass Index  
  - `children` — Number of dependents covered by insurance  
  - `smoker` — Smoking status (`yes`, `no`)  
  - `region` — Residential region (`northeast`, `northwest`, `southeast`, `southwest`)  
  - `charges` — Annual medical costs (target variable)

## 🛠️ Methodology

### 📊 Exploratory Data Analysis

- **Descriptive Statistics**: Basic counts, means, and missing values check  
- **Visualization**:
  - Bar charts to explore region counts and total charges  
  - Box plots examining sex, smoker, region vs. charges  
  - Scatter plots revealing age vs. charges trends

### 🏗️ Model Development

1. **Data Preprocessing**  
   - Numeric features: **MinMax Scaling** for `age`, `bmi`, and `children`  
   - Categorical features: **One-Hot Encoding** for `sex`, `smoker`, and `region`  
   - Target (`charges`): Also scaled for consistent error measurement  

2. **Multiple Regression Models**  
   - **12 Models** tested (Linear, Ridge, Lasso, Elastic Net, SGD, RandomForest, DecisionTree, **GradientBoosting**, AdaBoost, KNN, SVM, XGBoost)  
   - **Train/Test Split** at an 80/20 ratio

3. **Evaluation Metrics**  
   - 🏅 **RMSE** (Root Mean Squared Error)  
   - 🏅 **MSE** (Mean Squared Error)  
   - 🏅 **MAE** (Mean Absolute Error)  
   - 🏅 **R²** (Coefficient of Determination)

## 📈 Performance

- 🏆 **Gradient Boosting Regressor** achieved the best results before tuning  
- 🔧 **Hyperparameter Tuning** for `max_depth`, `n_estimators`, `learning_rate`, `subsample` significantly reduced overfitting  
- 🎯 **Tuned RMSE** on test data is approximately **0.069** (with scaled target), indicating high accuracy

## 💡 Key Contributions

- 🤝 **Holistic Approach**: End-to-end solution from EDA through deployment-ready model pipelines  
- 📊 **Insights into Health Factors**: Highlights how smoking, BMI, and age critically impact insurance charges  
- 🚀 **Efficient Pipelines**: Demonstrates a robust scikit-learn workflow for data preprocessing and model training

## 🔬 Limitations and Future Work

- 📊 **Data Size**: The dataset is relatively small, which may limit generalization  
- 🏷️ **Feature Engineering**: Additional non-linear transformations or domain-specific features could improve performance  
- 🌎 **Real-World Application**: Further validation required with larger, more diverse datasets

## 📚 References

1. **scikit-learn** – [https://scikit-learn.org](https://scikit-learn.org)  
2. **XGBoost Documentation** – [https://xgboost.readthedocs.io](https://xgboost.readthedocs.io)

## 🙏 Acknowledgements

A heartfelt thank you to the **open-source community** for the powerful libraries (NumPy, pandas, Matplotlib, scikit-learn, XGBoost) and to researchers who continually advance the field of machine learning. This project stands on the shoulders of these collective contributions.
