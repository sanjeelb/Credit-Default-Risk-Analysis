# Home Credit Default Risk Prediction


## Project Overview
This project focuses on predicting the likelihood of a borrower defaulting on a loan using machine learning techniques. The goal is to enhance financial inclusion while managing credit risk by analyzing demographic, behavioral, and transactional data from Home Credit. The project demonstrates the use of Logistic Regression, Random Forest, and XGBoost models to classify borrowers and evaluate default risk effectively.




## Business Objective
- Predict loan default risk for borrowers with limited or no credit history
- Identify key factors influencing creditworthiness
- Enable data-driven lending decisions to minimize financial losses
- Support Home Credit’s mission of providing accessible loans while managing risk




## Dataset Description
Source: Kaggle Home Credit Default Risk
Rows: ~3 million loan applications
Columns: 121 features
Target Variable: Target (1 = Default, 0 = Non-default)




## Data Preprocessing
- Removed columns with >30% missing values
- Imputed missing values using mean/median based on distribution skewness
- Encoded categorical variables using Label Encoding and mapping techniques
- Addressed multicollinearity via correlation analysis
- Applied SMOTE for handling class imbalance
- Split data into training and test sets for model evaluation




## Exploratory Data Analysis (EDA)
- Examined borrower demographics and financial behavior
- Analyzed correlations between features and target variable
- Visualized distributions of key variables and identified outliers
- Assessed feature importance to select meaningful predictors




##  Modeling Approach
Models implemented:
1. Logistic Regression
Achieved AUC of 0.60, showing limited predictive power
Useful as baseline model for comparison
2. Random Forest (Tuned)
Hyperparameter tuning improved AUC to 0.70
Captured more complex patterns than logistic regression
3. XGBoost (Tuned)
Achieved highest AUC of 0.73
Used GridSearchCV for hyperparameter optimization
Balanced model performance with reduced overfitting




## Key Insights
- Income, credit amount, annuity, and loan type significantly impact default risk
- Borrowers with unstable employment or low income are at higher risk
- Advanced ensemble methods outperform simpler models in capturing complex relationships




##  Future Scope
- Deep Learning Integration: Combine deep learning models with Random Forest for better handling of complex patterns.
- Improved Predictive Power: Multi-layer neural networks may boost ROC AUC by 5-10%.
- Target ROC AUC: Aim for 0.95+ by integrating ensemble and deep learning approaches.
- Enhanced Edge Case Handling: Improve predictions for moderate-confidence cases and adapt to varied data types.
