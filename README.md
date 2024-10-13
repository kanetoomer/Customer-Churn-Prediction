# **Customer Churn Prediction**

**Table of Contents:**

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis](#exploratory-data-analysis-eda)
4. [Feature Engineering](#feature-engineering)
5. [Modeling](#modeling)
6. [Evaluation](#evaluation)
7. [Conclusion](#conclusion)
8. [How to Run the Project](#how-to-run-the-project)
9. [References](#references)



## Project Overview

**Project Type**: Regression Analysis & Predictive Modeling

**Objective**: The goal of this project is to predict whether a customer will churn based on historical usage data, demographics, and behavior patterns.

Customer churn is a crucial metric for businesses that rely on subscription models, as it measures the loss of customers over time. The ability to accurately predict churn helps businesses retain customers and improve overall revenue.

**Key Steps**:

- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Feature engineering 
- Model building (logistic regression, decision trees, random forest)
- Model evaluation and interpretation

**Tools**
- Python 3
  - Pandas
  - matplotlib
  - seaborn
  - scikit-learn
- PyCharm
- Jupyter Notebook

## Dataset

**Source**: IBM Telco Customer Churn Dataset

**The dataset consists of the following features**:

- **Customer demographics**: Gender, SeniorCitizen, Partner, Dependents
- **Account information**: Tenure, Contract type, Payment method, Monthly charges
- **Service information**: Internet service, Online security, Tech support, Streaming TV
- **Target variable**: Churn (Yes/No)

**Data Preparation**:

- Handling missing values (e.g., ***TotalCharges***)
- Encoding categorical variables (Label Encoding)
- Scaling numerical features (Monthly Charges, Total Charges)



## Exploratory Data Analysis (EDA)

**EDA was performed to understand the relationships between customer behavior and churn**:

- **Churn distribution**: Visualizing the number of customers who have churned vs. those who have not.
- **Demographic analysis**: Examining customer age, tenure, and contract types to explore any trends related to churn.
- **Correlation heatmap**: Analyzing feature relationships and their impact on churn using correlation metrics.

**Key Findings**:

- Customers with shorter tenures are more likely to churn.
- Higher monthly charges are associated with higher churn rates.
- Customers with longer contract types (e.g., 1-year or 2-year contracts) are less likely to churn.



## Feature Engineering

Several feature engineering techniques were used to improve model performance:

- **Encoding categorical features**: Label encoding was applied to features like **Contract**, **InternetService**, and **PaymentMethod**.
- **Scaling**: Used **StandardScaler** to normalize continuous variables like **MonthlyCharges** and **TotalCharges**.



## Modeling

**The following models were built and compared**:

- **Logistic Regression**: A simple and interpretable model for binary classification.
- **Random Forest Classifier**: An ensemble method that reduces overfitting and improves performance by averaging multiple decision trees.

**Train-Test Split**: The data was split into training and testing sets (80/20) to validate model performance.



## Evaluation

**The models were evaluated using the following metrics**:

- **Accuracy**: The percentage of correct predictions.
- **Precision and Recall**: To balance false positives and false negatives.
- **F1-Score**: A harmonic mean of precision and recall.
- **ROC-AUC**: Evaluates the true positive rate vs. false positive rate.

**Confusion Matrix**: 
- Provides insight into false positives and false negatives for each model.

**Model Comparison**:
- ***Logistic Regression*** achieved an accuracy of 81.5% and an AUC score of 0.74.
- ***Random Forest*** achieved an accuracy of 79.6%, but showed better potential for capturing more complex patterns in the data.



## Conclusion

The project successfully demonstrated that customer churn can be predicted using historical data. The ***Logistic Regression*** model performed well with an ***AUC score of 0.74***, while the ***Random Forest*** model provided similar accuracy, but with potential improvements via hyperparameter tuning.

The most important features influencing churn were ***Contract Type***, ***Tenure***, and ***Monthly Charges***.

Understanding churn drivers enables businesses to design targeted strategies, such as offering long-term contracts or discounts for high-risk customers, to improve retention and reduce churn rates.



## How to Run the Project

### 1. ***Clone the repository***: 
```
git clone (https://github.com/kanetoomer/Customer-Churn-Prediction.git)
```
### 2. ***Install dependencies***:

***For Windows***:
```
pip install pandas matplotlib seaborn scikit-learn
```
***For MacOS***:
```
pip3 install pandas matplotlib seaborn scikit-learn
```

### 3. ***Run the Jupyter Notebook***:
```
jupyter notebook churn_prediction.ipynb
```



## References

- **Dataset**: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **scikit-learn Documentation**: [Click Here](https://scikit-learn.org)
- **Matplotlib Documentation**: [Click Here](https://matplotlib.org)
- **Flask Documentation**: [Click Here](https://flask.palletsprojects.com)
