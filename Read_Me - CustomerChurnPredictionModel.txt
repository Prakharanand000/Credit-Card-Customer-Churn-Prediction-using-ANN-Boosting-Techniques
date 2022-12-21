# Introduction
Customer churn (also known as customer attrition) occurs when a customer stops using a company's products or services. 
Customer churn affects profitability, especially in industries where revenues are heavily dependent on subscriptions (e.g. banks, telephone and internet service providers, pay-TV companies, insurance firms, etc.). It is estimated that acquiring a new customer can cost up to five times more than retaining an existing one.
Therefore, customer churn analysis is essential as it can help a business:
- identify problems in its services (e.g. poor quality product/service, poor customer support, wrong target audience, etc.), and 
- make correct strategic decisions that would lead to higher customer satisfaction and consequently higher customer retention.


Objective
The goal of this notebook is to understand and predict customer churn for a bank. Specifically, we will initially perform Exploratory Data Analysis (EDA) to identify and visualise the factors contributing to customer churn. This analysis will later help us build Machine Learning models to predict whether a customer will churn or not.
This problem is a typical classification task. The task does not specify which performance metric to use for optimising our machine learning models. I decided to use recall since correctly classifying elements of the positive class (customers who will churn) is more critical for the bank.
Skills: Exploratory Data Analysis, Data Visualisation, Data Preprocessing (Feature Selection, Encoding Categorical Features, Feature Scaling), Addressing Class Imbalance (SMOTE), Model Tuning.
Models Used: Logistic Regression, Support Vector Machines, Random Forests, Gradient Boosting, XGBoost, and Light Gradient Boosting Machine.


Conclusions
Our final report to the bank should be based on two main points:
1.) EDA can help us identify which features contribute to customer churn. Additionally, feature importance analysis can quantify the importance of each feature in predicting the likelihood of churn. Our results reveal that the most significant feature is age (older customers are more likely to churn), followed by the number of products (having more products increases a customer’s likelihood to churn). The bank could use our findings to adapt and improve its services in a way that increases satisfaction for those customers more likely to churn.
2.) We can build several machine learning models with recall approximately equal to 78%, meaning that they can successfully detect almost 80% of those customers more luckily to churn. Perhaps, adding more features or/and records could help us improve predictive performance. Therefore, the bank could benefit from investing in gathering more data.