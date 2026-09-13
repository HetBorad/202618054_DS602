# Medical Insurance Statistical Modeling & Interactive Dashboard

## 🚀 Live Demo

👉 **[Open the Dashboard](https://medical-insurance-statistical-dashboard.streamlit.app/)**

## 📌 Project Overview

This project applies statistical modeling and hypothesis testing to the Medical Insurance Costs dataset.

The project includes:
- Exploratory Data Analysis (EDA)
- Descriptive Statistics
- Hypothesis Testing
- Multiple Linear Regression using OLS
- Residual Diagnostics
- Interactive Streamlit Dashboard
- Medical Insurance Cost Prediction

## 📊 Dataset

Dataset: Medical Insurance Costs

### Features

- age
- sex
- bmi
- children
- smoker
- region
- charges

## 🎯 Objectives

1. Perform exploratory data analysis.
2. Calculate descriptive statistics.
3. Perform hypothesis tests.
4. Build a multiple linear regression model using OLS.
5. Interpret regression coefficients and statistical significance.
6. Perform residual diagnostics.
7. Calculate VIF to check multicollinearity.
8. Develop an interactive Streamlit dashboard.
9. Provide real-time medical charge predictions.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Statsmodels
- Plotly
- Streamlit
- Joblib

## 🔄 Project Workflow

Data Collection
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Descriptive Statistics
↓
Hypothesis Testing
↓
Multiple Linear Regression
↓
Model Diagnostics
↓
Model Saving
↓
Streamlit Dashboard
↓
Prediction & Diagnostics

## 📈 Data Analysis

### Descriptive Statistics

The dataset was analyzed using:
- Mean
- Median
- Standard Deviation
- IQR
- Skewness
- Kurtosis

### Key Findings

- Age is approximately symmetrically distributed.
- BMI is slightly positively skewed.
- Children is right-skewed.
- Medical charges are strongly right-skewed.
- Age has the strongest correlation with charges among the numerical variables.
- BMI has a weak positive correlation with charges.
- Children has a very weak correlation with charges.

## 🧪 Hypothesis Testing

### Hypothesis Test 1: Smokers vs Non-Smokers

The medical charges of smokers and non-smokers were compared.

**H₀:** There is no significant difference in medical charges between smokers and non-smokers.

**H₁:** There is a significant difference in medical charges between smokers and non-smokers.

Significance level:

α = 0.05

Shapiro-Wilk and Levene's tests were performed before selecting the final statistical test.

Since the normality assumptions were not satisfied, the Mann-Whitney U test was used.

**Result:**

p-value < 0.001

Therefore, H₀ was rejected.

There is a statistically significant difference in medical charges between smokers and non-smokers.

### Hypothesis Test 2: Region vs Charges

A One-Way ANOVA was used to compare medical charges across the four regions.

**H₀:** Mean medical charges are equal across all regions.

**H₁:** At least one region has a different mean medical charge.

**F-statistic:** 2.9696

**p-value:** 0.030893

Since p < 0.05, H₀ was rejected.

Therefore, there is a statistically significant difference in mean medical charges across the regions.

## 📉 Multiple Linear Regression

A Multiple Linear Regression model was developed using
`statsmodels.api.OLS`.

The target variable was:

`charges`

Predictors included:

- age
- bmi
- children
- sex
- smoker
- region

Categorical variables were converted into dummy variables.

### Model Performance

**R²:** 0.751

**Adjusted R²:** 0.749

The model explains approximately 75.1% of the variation in medical charges.

### Important Findings

The major statistically significant predictors include:

- Age
- BMI
- Children
- Smoking status
- Southeast region
- Southwest region

Smoking status has the strongest effect among the predictors.

Smokers have substantially higher predicted medical charges than non-smokers, after controlling for the other variables.

## 🔍 Model Diagnostics

The following diagnostic techniques were used:

- Residuals vs Fitted Values
- Q-Q Plot
- Omnibus Test
- Jarque-Bera Test
- Durbin-Watson Test
- Variance Inflation Factor (VIF)

### Diagnostic Findings

The residuals show some evidence of non-constant variance.

The Q-Q plot shows deviation from the reference line, indicating that residuals are not perfectly normally distributed.

The Durbin-Watson statistic is approximately 2.09, indicating no strong evidence of autocorrelation.

All VIF values are below 2, indicating no serious multicollinearity among the predictors.

## 🌐 Streamlit Dashboard

The Streamlit application contains three main sections:

### 1. Data Exploration

Users can:
- Filter the dataset
- View summary statistics
- Explore distributions
- View scatter plots
- View correlations

### 2. Hypothesis Testing Lab

Users can dynamically select:
- Categorical variables
- Numerical variables

The application automatically performs the appropriate statistical test and displays the p-value and decision.

### 3. Live Prediction & Diagnostics

Users can enter:

- Age
- BMI
- Number of children
- Sex
- Smoking status
- Region

The application provides:
- Predicted medical charge
- 95% prediction interval
- Residual diagnostics
- Q-Q plot
- Statistical diagnostic measures

## ✅ Conclusion

This project demonstrates the complete statistical modeling workflow using the Medical Insurance Costs dataset.

Exploratory analysis showed that medical charges are strongly right-skewed. Hypothesis testing found significant differences in charges between smokers and non-smokers and across regions.

The Multiple Linear Regression model achieved an R² of 0.751, showing that the selected variables explain a substantial portion of the variation in medical charges.

The project also demonstrates residual diagnostics, multicollinearity analysis, model persistence, and deployment through an interactive Streamlit dashboard.