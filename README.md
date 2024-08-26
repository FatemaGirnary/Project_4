# Heart Disease Prediction Project

## Project Outline
The goal of this project is to accurately predict whether a patient will experience heart failure based on the following features:

- Age
- Sex
- ChestPainType
- RestingBP
- Cholesterol
- FastingBS
- RestingECG
- MaxHR
- ExerciseAngina
- Oldpeak
- ST_Slope
- HeartDisease

## Dataset
The dataset used contains 11 clinical features for predicting heart disease events.

- **Total Observations:** 1,190
- **Duplicated Observations:** 272
- **Final Dataset:** 918 observations

[Heart Failure Prediction Dataset (Kaggle)](https://www.kaggle.com/)

## Preprocessing
The project utilized the Scikit-learn library for the machine learning workflow, from data preprocessing to model evaluation. PySpark was also used to read and inspect the data, while Pandas was employed for primary data manipulation.

### Preprocessing Steps:
- The dataset was split into features (X) and the target variable (y), representing the presence of heart disease.
- The data was divided into training and testing sets using the `train_test_split` method from Scikit-learn with a defined `random_state` to ensure reproducibility.
- The number of input features was determined and printed as part of this process.

## Machine Learning Models
Since this is a binary classification problem (predicting whether heart failure occurs or not), the following Machine Learning Models were used:

### 1. Logistic Regression
- **Training:** The model was trained on labeled data (X_train, y_train) using the ‘liblinear’ solver with a set `random_state` for reproducibility and a maximum of 200 iterations.
- **Performance:** 
  - Confusion Matrix: Balanced true positives (122) and true negatives (84) with some misclassifications (13 false positives and 11 false negatives).
  - Classification Report: High precision (0.88-0.90), recall (0.87-0.92), and F1-scores (0.87-0.91) across both classes.
  - Final Accuracy: 90%

### 2. Random Forest
- **Performance:**
  - Class 0 (Healthy): Precision 0.88, Recall 0.86, F1-Score 0.87
  - Class 1 (Diseased): Precision 0.90, Recall 0.92, F1-Score 0.91
  - Overall Accuracy: 89%
  - Macro and weighted averages for precision, recall, and F1-scores consistently at 0.89.

### 3. XGBoost
- **Performance:**
  - Class 0 (Healthy): Precision 0.90, Recall 0.88, F1-Score 0.89
  - Class 1 (Diseased): Precision 0.91, Recall 0.93, F1-Score 0.92
  - Overall Accuracy: 91%
  - After removing the three lowest-ranked features, performance slightly declined, indicating the importance of even lower-ranked features.

### 4. Neural Networks
- **Configuration:**
  - Five hidden layers with a decreasing number of nodes (128 to 16) and sigmoid activation functions.
  - Initially used `binary_crossentropy` loss, later switched to `huber` loss, both with the `adam` optimizer.
  - Trained for 100 epochs with a 15% validation split.
  - Training accuracy improved, but validation accuracy remained at 49%.
  - Final Test Set Accuracy: 89%

## Deployment
The project includes a heart disease prediction app built with Streamlit. The app loads four pre-trained machine learning models (Logistic Regression, XGBoost, Random Forest, and Neural Networks) to predict heart disease based on user input.

### App Features:
- Users can input various health metrics (age, blood pressure, cholesterol, etc.) through the app's interface.
- The models generate predictions on the likelihood of heart disease.
- The app offers a user-friendly tool for heart disease risk assessment by leveraging multiple machine learning models to enhance prediction accuracy.

## Clinical Relevance

### Healthcare Applications of Predictive Models
Predictive models such as Random Forest and XGBoost play a crucial role in the early detection and risk assessment of heart disease. These models enable the identification of high-risk patients at an earlier stage, allowing for timely interventions and the development of personalized care plans.

### Minimizing Invasive Procedures
Invasive screening methods like angiograms and stress tests are often costly and risky. By leveraging machine learning models, the need for these procedures can be reduced, focusing instead on patients who are most likely to benefit from further evaluation.

### Enhancing Healthcare Outcomes
These predictive models enhance healthcare by optimizing resource allocation and improving patient outcomes. They provide data-driven insights that lead to more accurate diagnoses and better overall care.
