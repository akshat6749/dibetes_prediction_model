# 🩺 Comprehensive Diabetes Prediction Project

This repository contains a complete Python script for a machine learning project focused on predicting diabetes.
It follows the full end-to-end data science workflow — from data cleaning and EDA to model comparison, hyperparameter tuning, and final evaluation.

---

## 📌 1. Project Objective

The goal of this project is to develop a **high-accuracy machine learning model** to predict the onset of diabetes using health-related features.
The model can help identify at-risk individuals early and support preventive healthcare decisions.

---

## 📂 2. Dataset Information

**File:** `diabetes_prediction_dataset.csv`
**Source:** *(Add the dataset source, e.g., Kaggle)*
**Size:** 100,000 patient records

### **Features:**

| Feature             | Description                             |
| ------------------- | --------------------------------------- |
| gender              | Female, Male, Other                     |
| age                 | Patient age                             |
| hypertension        | 0 = No, 1 = Yes                         |
| heart_disease       | 0 = No, 1 = Yes                         |
| smoking_history     | Smoking status                          |
| bmi                 | Body Mass Index                         |
| HbA1c_level         | Hemoglobin A1c (indicator for diabetes) |
| blood_glucose_level | Current blood glucose                   |
| diabetes            | Target variable (0 = No, 1 = Yes)       |

---

## 🚀 3. Project Workflow

The `diabetes_project.py` script performs the following steps:

1. **Setup & Imports** – Load all necessary libraries.
2. **Data Loading & Inspection** – Read the dataset and examine structure.
3. **Data Cleaning** – Remove inconsistencies (e.g., remove gender “Other”).
4. **Exploratory Data Analysis (EDA)** – Generate visual insights.
5. **Preprocessing** –

   * Standard scaling for numeric features
   * One-hot encoding for categorical features
   * Pipeline built with scikit-learn
6. **Baseline Model Comparison** –
   Models compared: KNN, SVM, Logistic Regression, Decision Tree, Random Forest
   With & without PCA
7. **Hyperparameter Tuning** –
   GridSearchCV on the best model (Random Forest)
8. **Final Evaluation** –
   Confusion matrix, Classification Report, ROC-AUC curve
9. **Save Outputs** – All plots saved as `.png` files.

---

## 📈 4. Running the Project

### **Clone the repository**

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### **Install dependencies**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### **Add the dataset**

1. Download `diabetes_prediction_dataset.csv`
2. Place it in the same directory as `diabetes_project.py`

### **Run the script**

```bash
python diabetes_project.py
```

---

## 🧠 5. Key Findings & Results

### **EDA Insights**

* Dataset is **highly imbalanced** (~8.5% diabetic).
* **HbA1c_level** and **blood_glucose_level** are the strongest predictors.
* **Age**, **hypertension**, and **heart disease** correlate strongly with diabetes.

### **Model Comparison**

* Ensemble models (especially **Random Forest**) performed best.
* Best baseline ROC-AUC ≈ **0.96**.
* PCA did **not** improve model performance.
* KNN performed poorly due to high-dimensional encoded features.

### **Final Tuned Model: Random Forest**

| Metric       | Score   |
| ------------ | ------- |
| **Accuracy** | ~97.05% |
| **ROC-AUC**  | ~0.975  |

The model shows excellent predictive power with strong precision and recall for diabetic cases.


---

## 📜 License

Add your project license here (e.g., MIT License).

---

## 🤝 Contributing

Pull requests are welcome! Feel free to open an issue for major changes or feature suggestions.

---

