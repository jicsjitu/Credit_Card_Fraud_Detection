# 💳 Credit Card Fraud Detection Using Machine Learning

## Overview
This project aims to build a machine-learning model to detect fraudulent credit card transactions using Python. The dataset contains transactions made by credit cards in September 2013 by European cardholders. The dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. 

## Objective
The primary goal is to identify fraudulent credit card transactions based on a set of features like transaction amount, type of transaction, and time of transaction. The model is expected to achieve high accuracy in detecting these fraudulent activities to minimize financial losses for credit card companies and customers.

## Dataset
The dataset used for this project contains anonymized credit card transactions labeled as fraudulent or genuine. It includes the following features:
- Time: Time elapsed between each transaction and the first transaction in seconds.
- V1-V28: Principal components obtained with PCA (features are anonymized due to confidentiality reasons).
- Amount: Transaction amount.
- Class: 1 for fraudulent transactions, 0 otherwise.

## Methodology
i. **Data Preprocessing:** 
   - Exploratory Data Analysis (EDA) to understand the distribution of features.
   - Handling missing values and outliers.
   - Feature scaling and normalization.

ii. **Model Selection:**
   - Comparing the performance of various machine learning algorithms such as Logistic Regression, Random Forest, and Gradient Boosting.
   - Cross-validation and hyperparameter tuning to optimize model performance.

iii. **Model Evaluation:**
   - Evaluation metrics used: accuracy, precision, recall, F1-score, and ROC AUC.
   - Confusion matrix analysis to understand the model's performance in detecting fraudulent and non-fraudulent transactions.

iv. **Deployment:**
   - Creating a Python script for real-time prediction of fraudulent transactions.
   - Utilizing Streamlit for creating a basic web application (optional).

## Requirements
- Python 3.x
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, etc.

## The Web Interface
![WhatsApp Image 2024-06-25 at 19 48 06_906e62ca](https://github.com/jicsjitu/Credit_Card_Fraud_Detection/assets/162569175/a84a0dcc-3f13-429e-9320-4c3d2e222e6a)

## Testing Model

i. **For Legitimate Transaction**

![WhatsApp Image 2024-06-25 at 19 54 45_24b51f89](https://github.com/jicsjitu/Credit_Card_Fraud_Detection/assets/162569175/f428fd37-a311-4382-8b47-ddcc0be36d58)

ii. **For Fraudulent Transaction**

![WhatsApp Image 2024-06-25 at 19 55 48_a26acf31](https://github.com/jicsjitu/Credit_Card_Fraud_Detection/assets/162569175/bf5de623-80b9-4bab-8322-305384be583f)

## Usage
1. Clone the repository:
   ```
   git clone https://github.com/jicsjitu/credit_card_fraud_detection.git
   ```
2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebooks to train models and evaluate results.

## Files in the Repository

i. **README.md**: This file provides an overview of the project, setup instructions, usage guidelines, and other relevant information about the credit card fraud detection project using Python and machine learning.

ii. **Visualizations Folder**: Contains visual representations of data insights and model performance. This includes plots such as histograms, scatter plots, ROC curves, and confusion matrices generated during the exploratory data analysis and model evaluation phases.

iii. **app.py**: This Python script serves as the main application file for real-time prediction of fraudulent credit card transactions. It utilizes the trained machine learning model (stored in `best_model.pkl`) to classify transactions as fraudulent or genuine based on input data.

iv. **creditcard.csv**: The dataset file used for training and testing the machine learning models. It contains anonymized credit card transaction data, including features like transaction amount, time, and anonymized principal components (V1-V28), along with labels indicating whether each transaction is fraudulent (1) or genuine (0).

v. **best_model.pkl**: This file stores the serialized version of the best-performing machine learning model trained on the `creditcard.csv` dataset. The model is ready for deployment and can be loaded into `app.py` for making predictions on new credit card transactions.

vi. **credit_card_fraud_detection.ipynb**: Jupyter notebook containing the complete code for data preprocessing, model training, and evaluation.

vii. **requirements.txt**: List of libraries required to run the code.

## Credits
- Dataset: [[Credit Card](https://drive.google.com/file/d/1u_9Zr5cEZYSCn-YhG4Ymrct6oHcNKTNC/view?usp=sharing)]

## Feedback and Questions
If you have any feedback or questions about the project, please feel free to ask. I am here to help and appreciate your input. You can reach out by opening an issue on GitHub or by emailing me at jitukumar9387@gmail.com

Thank you for checking out the Credit Card Fraud Detection Project! We hope you find it useful and informative.

Happy analyzing!

