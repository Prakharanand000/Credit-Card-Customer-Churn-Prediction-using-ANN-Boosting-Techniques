# Customer Churn Prediction with Machine Learning 🏦💰

A comprehensive machine learning project for predicting bank customer churn using advanced classification algorithms and extensive exploratory data analysis. This project demonstrates end-to-end ML workflows including EDA, data preprocessing, feature engineering, model tuning, and performance evaluation.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red.svg)](https://xgboost.readthedocs.io/)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Key Features](#key-features)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Models](#machine-learning-models)
- [Key Findings](#key-findings)
- [Results](#results)
- [Technologies Used](#technologies-used)

## 🎯 Project Overview

This project tackles the critical business challenge of **customer churn prediction** in the banking sector. Using a comprehensive dataset of 10,000 bank customers, we perform in-depth exploratory data analysis to identify churn patterns and build multiple machine learning models to predict which customers are likely to leave the bank.

**Primary Objective:** Build a classification model that accurately identifies customers at risk of churning, enabling the bank to implement targeted retention strategies.

**Optimization Metric:** **Recall** - Maximizing the detection of customers who will actually churn is more critical for the bank than overall accuracy, as missing a churning customer is costlier than false positives.

## 💼 Business Problem

Customer churn represents a significant challenge for banks, as:
- Acquiring new customers is 5-25x more expensive than retaining existing ones
- Churning customers represent lost revenue and relationship value
- Understanding churn drivers enables targeted retention strategies
- Predictive models allow proactive intervention before customers leave

This project addresses these challenges by:
1. Identifying key factors contributing to customer churn
2. Building predictive models to detect at-risk customers
3. Providing actionable insights for retention strategies

## 📊 Dataset

**Source:** Bank Customer Churn Dataset  
**Size:** 10,000 customers × 14 features  
**Target Variable:** `Exited` (0 = Retained, 1 = Churned)

### Features:
- **Customer Information:**
  - `CustomerId`: Unique customer identifier
  - `Surname`: Customer surname
  - `Age`: Customer age (18-92)
  - `Geography`: Customer location (France, Spain, Germany)
  - `Gender`: Male/Female

- **Banking Relationship:**
  - `Tenure`: Years with the bank (0-10)
  - `Balance`: Account balance
  - `NumOfProducts`: Number of bank products (1-4)
  - `HasCrCard`: Credit card holder (0/1)
  - `IsActiveMember`: Active membership status (0/1)

- **Financial Information:**
  - `CreditScore`: Customer credit score
  - `EstimatedSalary`: Customer salary estimate

- **Target:**
  - `Exited`: Churn status (0 = No, 1 = Yes)

### Data Characteristics:
- **No missing values** - Clean dataset ready for analysis
- **Class imbalance** - 80% retained vs 20% churned (addressed using SMOTE)
- **Mixed data types** - Continuous, categorical, and binary features

## ✨ Key Features

### Technical Highlights:
- ✅ Comprehensive Exploratory Data Analysis (EDA) with statistical testing
- ✅ Advanced feature engineering and selection
- ✅ Class imbalance handling using SMOTE
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ 6 different ML algorithms with ensemble methods
- ✅ Cross-validation for robust performance estimation
- ✅ Multiple evaluation metrics (Accuracy, Precision, Recall, AUC)
- ✅ Learning curves for overfitting analysis
- ✅ Feature importance analysis
- ✅ ROC curves and Cumulative Gain charts

### Skills Demonstrated:
- Data Analysis & Visualization
- Statistical Testing (Chi-square)
- Feature Engineering
- Data Preprocessing & Scaling
- Class Imbalance Techniques
- Hyperparameter Optimization
- Model Evaluation & Comparison
- Ensemble Learning

## 🔍 Exploratory Data Analysis

### Target Variable Distribution
- **Retained Customers:** 79.63% (6,354 customers)
- **Churned Customers:** 20.37% (1,627 customers)
- **Class Imbalance:** Significant (addressed with SMOTE)

### Continuous Variables Analysis

#### 1. **Age** 🎂
- **Key Finding:** Older customers significantly more likely to churn
- **Distribution:** Mean age ≈ 40 years, range 18-92
- **Insight:** Clear difference between age groups - older demographics need targeted retention

#### 2. **Credit Score** 💳
- **Key Finding:** No significant difference between churned/retained customers
- **Distribution:** Most values above 600
- **Insight:** Credit score is not a strong predictor of churn

#### 3. **Balance** 💰
- **Key Finding:** Similar distributions for both groups
- **Distribution:** Roughly normal (excluding zero balances)
- **Insight:** Account balance alone doesn't strongly indicate churn risk

#### 4. **Estimated Salary** 💵
- **Key Finding:** Uniform distribution, no significant effect on churn
- **Insight:** Salary level is not a churn predictor

### Categorical Variables Analysis

#### 1. **Geography** 🌍
- **Key Finding:** German customers have ~2x churn rate vs France/Spain
- **France:** 50.56% of customers, lower churn
- **Germany:** 25.10% of customers, **highest churn rate**
- **Spain:** 24.34% of customers, lower churn
- **Insight:** Geographic-specific factors drive churn (competition, preferences)

#### 2. **Gender** 👥
- **Key Finding:** Female customers more likely to churn
- **Male Customers:** 54.52%
- **Female Customers:** 45.48%, higher churn rate
- **Insight:** Gender-specific banking needs may not be met

#### 3. **Tenure** 📅
- **Key Finding:** No significant effect on churn rate
- **Distribution:** Fairly uniform (1-10 years)
- **Insight:** Relationship length alone doesn't prevent churn

#### 4. **Number of Products** 📦
- **Key Finding:** Having 3-4 products dramatically increases churn!
- **1-2 Products:** Normal churn rate
- **3-4 Products:** Significantly higher churn
- **Insight:** Bank may struggle to support multi-product customers

#### 5. **Credit Card** 💳
- **Key Finding:** No effect on churn rate
- **70.56%** of customers have credit cards
- **Insight:** Credit card ownership not a retention factor

#### 6. **Active Membership** ⚡
- **Key Finding:** Inactive customers much more likely to churn
- **Active Members:** 51.51%
- **Inactive Members:** Higher churn rate
- **Insight:** Customer engagement is critical for retention

### Statistical Validation
- **Chi-square tests** performed on categorical variables
- Features with p-value > 0.05 identified as non-significant
- Correlation analysis showed no multicollinearity issues

## 🔧 Data Preprocessing

### 1. Feature Selection
**Dropped Features:**
- `RowNumber`, `CustomerId`, `Surname` - Customer-specific identifiers
- `EstimatedSalary` - Uniform distribution, no predictive power
- `Tenure` - No significant effect on churn (p-value > 0.05)
- `HasCrCard` - No significant effect on churn (p-value > 0.05)

**Retained Features:** 7 informative features for modeling

### 2. Encoding Categorical Features

**Label Encoding** (Gender):
```python
Gender: Male → 1, Female → 0
```

**Custom Encoding** (Geography):
```python
Geography: Germany → 1, France/Spain → 0
```
*Rationale: Germany shows distinct churn pattern; France and Spain have similar rates*

### 3. Feature Scaling
- **Method:** StandardScaler (zero mean, unit variance)
- **Scaled Features:** `CreditScore`, `Age`, `Balance`
- **Purpose:** Normalize feature ranges for distance-based algorithms (SVM, etc.)

### 4. Addressing Class Imbalance
- **Problem:** 80% Retained vs 20% Churned
- **Solution:** SMOTE (Synthetic Minority Oversampling Technique)
- **Result:** Balanced classes for unbiased model training
- **Implementation:** `imblearn.over_sampling.SMOTE`

**Before SMOTE:**
- Class 0 (Retained): 6,354
- Class 1 (Churned): 1,627

**After SMOTE:**
- Class 0 (Retained): 6,354
- Class 1 (Churned): 6,354

## 🤖 Machine Learning Models

### Baseline Models
Initial performance benchmarking:

| Model | Recall |
|-------|--------|
| Gaussian Naïve Bayes | ~70% |
| Logistic Regression | ~70% |

### Tuned Classification Models

All models optimized using **GridSearchCV** with 5-fold cross-validation, optimizing for **Recall**.

#### 1. **Logistic Regression** 📈
- **Hyperparameters Tuned:** C, penalty, solver, max_iter
- **Best Config:** C=10, penalty=l2, solver=lbfgs
- **Strengths:** Interpretable, fast training
- **Use Case:** Baseline comparison, coefficient interpretation

#### 2. **Support Vector Classifier (SVC)** 🎯
- **Hyperparameters Tuned:** kernel, C, gamma
- **Best Config:** RBF kernel, C=2, gamma=scale
- **Strengths:** Effective in high-dimensional spaces
- **Use Case:** Non-linear decision boundaries

#### 3. **Random Forest** 🌲
- **Hyperparameters Tuned:** n_estimators, max_depth, min_samples_split/leaf, criterion
- **Best Config:** 100 estimators, max_depth=6
- **Strengths:** Feature importance, robust to overfitting
- **Use Case:** Handling non-linear relationships, feature analysis

#### 4. **Gradient Boosting Classifier** 🚀
- **Hyperparameters Tuned:** n_estimators, learning_rate, max_depth, subsample
- **Best Config:** 600 estimators, learning_rate=0.01
- **Strengths:** Sequential error correction
- **Use Case:** High performance, handles complex patterns

#### 5. **XGBoost Classifier** ⚡
- **Hyperparameters Tuned:** learning_rate, max_depth, reg_alpha/lambda, subsample, colsample_bytree, gamma
- **Best Config:** Optimized regularization parameters
- **Strengths:** **Highest Recall (78.5%)**, efficient, regularization
- **Use Case:** Best performer for churn detection

#### 6. **LightGBM Classifier (LGBM)** 💡
- **Hyperparameters Tuned:** max_depth, num_leaves, learning_rate, feature_fraction, reg_alpha/lambda
- **Best Config:** Balanced complexity and performance
- **Strengths:** **Best Overall Performance** (Accuracy, Precision, AUC)
- **Use Case:** Balanced performance across all metrics

### Ensemble Learning

#### **Soft Voting Classifier** 🗳️
- **Combines:** SVC, Random Forest, Gradient Boosting, XGBoost, LGBM
- **Method:** Weighted probability averaging
- **Performance:** Strong generalization
- **Recall:** ~78%

## 📈 Key Findings

### Most Important Features (by importance):
1. **Age** ⭐⭐⭐⭐⭐ - Most critical predictor
2. **NumOfProducts** ⭐⭐⭐⭐ - Strong indicator
3. **IsActiveMember** ⭐⭐⭐ - Important engagement metric
4. **Balance** ⭐⭐ - Moderate importance
5. **Geography** ⭐ - Regional differences
6. **Gender** ⭐ - Minor effect
7. **CreditScore** - Least important

### Business Insights:

#### 🎯 **High-Risk Customer Profile:**
- **Age:** 40+ years old
- **Location:** Germany
- **Gender:** Female
- **Products:** 3-4 bank products
- **Activity:** Inactive member
- **Action:** Priority retention targeting

#### 📊 **Churn Drivers:**
1. **Age:** Older customers (45+) show significantly higher churn
2. **Geographic Location:** German customers churn at 2x rate
3. **Product Complexity:** 3-4 products → higher churn (service quality issues?)
4. **Engagement:** Inactive members highly likely to churn
5. **Gender:** Female customers show higher churn propensity

#### 💡 **Retention Strategies:**
- **Senior Customer Programs:** Tailored services for 45+ demographic
- **Germany-Specific Initiatives:** Address regional competition/preferences
- **Multi-Product Support:** Improve service for customers with 3+ products
- **Engagement Campaigns:** Convert inactive to active members
- **Female-Focused Services:** Address specific banking needs

## 📊 Results

### Model Performance Comparison

#### Training Set Performance (5-Fold CV):

| Model | Accuracy | Precision | Recall | AUC |
|-------|----------|-----------|--------|-----|
| Logistic Regression | 73.2% | 73.5% | 72.8% | 0.810 |
| SVC | 76.1% | 76.3% | 76.0% | 0.842 |
| Random Forest | 79.2% | 79.8% | 78.5% | 0.875 |
| Gradient Boosting | 80.1% | 80.5% | 79.6% | 0.882 |
| **XGBoost** | 80.5% | 79.9% | **78.5%** ⭐ | 0.885 |
| **LightGBM** | **81.0%** ⭐ | **81.2%** ⭐ | 78.3% | **0.888** ⭐ |
| Soft Voting | 80.3% | 80.7% | 78.0% | 0.883 |

#### Test Set Performance:

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Logistic Regression | 72.8% | 73.1% | 72.5% |
| SVC | 75.9% | 76.1% | 75.7% |
| Random Forest | 78.8% | 79.2% | 78.1% |
| Gradient Boosting | 79.7% | 80.1% | 79.2% |
| XGBoost | 80.2% | 79.7% | 78.1% |
| LightGBM | 80.6% | 80.9% | 77.9% |
| Soft Voting | 79.9% | 80.3% | 77.6% |

### Key Performance Insights:

✅ **No Overfitting:** Training and test performance very similar  
✅ **Strong Generalization:** Models perform consistently on unseen data  
✅ **High Recall:** Successfully detect **~78%** of churning customers  
✅ **Balanced Performance:** High accuracy and precision maintained  

### Learning Curves Analysis:
- **Small gap** between training/validation curves at end of training
- Indicates proper model complexity (not overfitting)
- More data unlikely to significantly improve performance

### ROC Curve Analysis:
- **LGBM achieves highest AUC (0.888)**
- All models significantly better than random classifier
- Trade-off between True Positive Rate and False Positive Rate well-balanced

### Cumulative Gains Chart:
**Key Metric:** Targeting 50% of customers most likely to churn → Model captures **80%** of actual churners
- **Business Value:** 60% more efficient than random selection
- **Application:** Focus retention resources on high-probability churners

## 🛠️ Technologies Used

### Core Libraries:
- **Python 3.x** - Programming language
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Visualization
- **Seaborn** - Statistical visualization

### Machine Learning:
- **scikit-learn** - ML algorithms and utilities
  - Train/test splitting
  - Cross-validation
  - Preprocessing (StandardScaler, LabelEncoder)
  - Models (Logistic Regression, SVC, Random Forest, Gradient Boosting)
  - Metrics (accuracy, precision, recall, ROC-AUC)
  - GridSearchCV for hyperparameter tuning
  
- **XGBoost** - Extreme Gradient Boosting
- **LightGBM** - Light Gradient Boosting Machine
- **imbalanced-learn** - SMOTE for class imbalance

### Statistical Analysis:
- **SciPy** - Statistical tests (chi-square)
- **scikit-plot** - Visualization tools (cumulative gains)

### Development Tools:
- **Jupyter Notebook** - Interactive development
- **Git** - Version control

### Prerequisites:
- Python 3.7 or higher
- pip package manager

### Contribution Ideas:
- Add new features or algorithms
- Improve documentation
- Create additional visualizations
- Optimize code performance
- Add unit tests
- Implement deployment solutions

## 📚 References

1. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** by Aurélien Géron
2. **Feature Engineering and Selection: A Practical Approach for Predictive Models** by Max Kuhn and Kjell Johnson
3. **SMOTE: Synthetic Minority Over-sampling Technique** - Chawla et al.
4. **XGBoost: A Scalable Tree Boosting System** - Chen & Guestrin
5. scikit-learn Documentation: https://scikit-learn.org/
6. XGBoost Documentation: https://xgboost.readthedocs.io/
7. LightGBM Documentation: https://lightgbm.readthedocs.io/

---

⭐ **If you found this project helpful, please consider giving it a star!**


**Made with ❤️ and Python**
