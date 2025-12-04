# 🔥 Fire Weather Index (FWI) Prediction — Ridge Regression Model

## 📘 Project Overview
This project predicts the **Fire Weather Index (FWI)** using meteorological and environmental data through **Ridge Regression**.  
It includes data preprocessing, feature engineering, model training, evaluation, and saving the trained model for deployment.

---

## 🧩 Repository Structure
```
FWI-Predictor/
├── fwi_dataset_asitwas.csv
├── fwi_predictor_nb.ipynb
├── ridge.pkl
├── scaler.pkl
├── static
│ └── style.css
├── templates
│ ├── index.html
│ └── home.html
└──  requirements.txt

```



---

## 🎯 Objective
To develop a regression-based model that accurately predicts the **Fire Weather Index (FWI)** using weather and fire danger indices.

---

## ⚙️ Key Features
- Data cleaning, preprocessing, and feature selection  
- Handling of missing values and outliers  
- Exploratory data analysis (EDA) using plots and correlation matrices  
- Model training using **Ridge Regression**  
- Performance evaluation using **R²**, **MAE**, and **RMSE**  
- Model saving for deployment (`ridge.pkl` and `scaler.pkl`)  

---

## 🤖 Model Summary
- **Algorithm Used:** Ridge Regression  
- **Target Variable:** Fire Weather Index (FWI)  
- **Performance Metrics:**
  - R² Score: **0.982**
  - MAE: **0.58**
  - RMSE: **0.74**
- **Optimal Alpha:** 1 (determined via GridSearchCV)

---

## 💻 Installation & Setup

### Prerequisites
- Python **3.10**

### Steps
```bash
# Clone the repository
git clone https://github.com/MahiiBamba/ml-projects.git
cd ml-projects/FWI-Predictor

# Install dependencies
pip install -r requirements.txt
```
## 📦 Requirements
  - pandas==2.2.2
  - numpy==1.26.4
  - matplotlib==3.9.2
  - seaborn==0.13.2
  - scikit-learn==1.5.1

## WorkFlow summary:
  - "Data Collection: Loaded and verified raw dataset integrity, converted labels, and checked for type mismatches."
  - "Data Preprocessing: Handled missing values and outliers, cleaned and encoded categorical columns, dropped irrelevant time-based features."
  - "Feature Engineering & Scaling: Selected key predictors correlated with FWI and normalized features using StandardScaler."
  - "Model Training (Ridge Regression): Trained with L2 regularization to handle multicollinearity and achieved high predictive performance (R² = 0.982)."
  - "Evaluation & Optimization: Used MAE, RMSE, and R² metrics; grid search confirmed alpha = 1 as optimal."

## 📈 Outputs:

  - Trained Model: ridge.pkl
  - Scaler File: scaler.pkl
  - Cleaned Dataset: cleaned_dataset.csv
  - Documentation: fwi_Doc_(Mansi_Bambal).docx

## How to run notebook:

  - Open Jupyter Notebook or VS Code.
  - Load `fwi_predictor(Mansi_Bambal).ipynb`.
  - Run cells sequentially from top to bottom to replicate data preprocessing, model training, and evaluation results.

