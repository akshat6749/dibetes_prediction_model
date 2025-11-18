Comprehensive Diabetes Prediction Project

This repository contains a complete Python script for a machine learning project focused on predicting diabetes. The project follows a full data science workflow, from data cleaning and exploratory data analysis (EDA) to model comparison, hyperparameter tuning, and final evaluation.

1. Project Objective

The primary objective of this project is to develop a high-accuracy machine learning model to predict the onset of diabetes based on a set of health-related features. This model aims to serve as a tool to identify at-risk individuals, enabling early intervention and preventative care.

2. The Dataset

File: diabetes_prediction_dataset.csv

Source: (You can add the source here, e.g., Kaggle)

Size: 100,000 patient records

Features

The dataset includes 9 features:

gender: Patient's gender (Female, Male, Other)

age: Patient's age

hypertension: 0 (No) or 1 (Yes)

heart_disease: 0 (No) or 1 (Yes)

smoking_history: Patient's smoking status

bmi: Body Mass Index

HbA1c_level: Hemoglobin A1c level (a key diabetes indicator)

blood_glucose_level: Current blood glucose level

diabetes: Target variable, 0 (No) or 1 (Yes)

3. Project Workflow

The diabetes_project.py script follows these steps:

Setup & Imports: Load all necessary libraries.

Data Loading & Initial Inspection: Load the data and perform a preliminary examination.

Data Cleaning & Filtering: Address inconsistencies (e.g., remove 'Other' gender for low representation).

Exploratory Data Analysis (EDA): Deeply investigate the data with visualizations to find patterns.

Data Preprocessing: Prepare the data for modeling using a scikit-learn pipeline (StandardScaler for numerical, OneHotEncoder for categorical).

Baseline Model Comparison: Train and evaluate multiple models (KNN, SVM, Logistic Regression, Random Forest, Decision Tree) with and without PCA.

Hyperparameter Tuning: Select the most promising model (Random Forest) and tune it using GridSearchCV.

Final Model Evaluation: Thoroughly evaluate the best model on the hold-out test set using a classification report, confusion matrix, and ROC-AUC curve.

Generate Outputs: Save all EDA and evaluation plots as .png files.

4. How to Run This Project

Clone the repository:

git clone [https://github.com/](https://github.com/)[your-username]/[your-repo-name].git
cd [your-repo-name]


Install the required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn


Add your data:

Download the diabetes_prediction_dataset.csv file.

Place it in the same directory as the diabetes_project.py script.

Run the script:

Open your terminal or command prompt.

Navigate to the project folder.

Execute the script:

python diabetes_project.py


5. Key Findings & Results

Exploratory Data Analysis (EDA):

The dataset is highly imbalanced, with only ~8.5% of patients having diabetes. This makes ROC-AUC a more reliable metric than accuracy.

HbA1c_level and blood_glucose_level were found to be the strongest predictors of diabetes.

age, hypertension, and heart_disease also showed a strong positive correlation with the diabetes outcome.

Model Comparison:

Ensemble methods (Random Forest, Gradient Boosting) were the clear top performers.

Random Forest was selected as the best baseline model with a Mean ROC-AUC score of ~0.96.

PCA (Principal Component Analysis) did not improve performance and was not used in the final model.

KNN was the worst-performing model, likely due to the high dimensionality created by one-hot encoding.

Final Model Performance:

Model: Tuned Random Forest Classifier

Final Accuracy: ~97.05%

Final ROC-AUC: ~0.975

The final model performs exceptionally well, with a high F1-score for the positive (Diabetic) class, indicating a good balance between precision and recall.

6. Generated Output Files

When you run diabetes_project.py, the following image files will be saved in the root directory:

01_target_distribution.png: Bar chart of the imbalanced target variable.

02_numerical_distributions.png: Histograms for age, bmi, HbA1c_level, and blood_glucose_level.

03_numerical_vs_diabetes.png: Density plots showing how numerical features differ for diabetic vs. non-diabetic patients.

04_correlation_matrix.png: A heatmap of correlations between features.

05_model_comparison.png: A bar chart comparing the ROC-AUC scores of all baseline models.

06_confusion_matrix.png: The final confusion matrix for the tuned model on the test set.

07_roc_auc_curve.png: The final ROC curve, showing the excellent tradeoff between True Positives and False Positives.
