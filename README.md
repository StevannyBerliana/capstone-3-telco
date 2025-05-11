Telco Customer Churn Prediction
Project Overview
This project focuses on predicting customer churn in the telecommunications industry using machine learning techniques. The goal is to identify customers at high risk of churning and understand the key factors driving churn, enabling the company to implement targeted retention strategies. The dataset includes customer information such as tenure, contract type, monthly charges, and service usage, with the target variable being whether a customer churned (Yes/No).
Problem Statement
The telecommunications industry faces intense competition, leading to customer churn due to dissatisfaction with service quality, pricing, or support. Churn results in revenue loss and higher costs for acquiring new customers. This project aims to build a predictive model to identify at-risk customers and provide insights into churn drivers to enhance retention efforts.
Objectives

Develop a machine learning model to predict customers likely to churn.
Identify key factors contributing to churn to inform retention strategies.
Minimize false negatives (failing to predict churn) to reduce revenue loss.
Achieve cost savings by targeting retention efforts effectively.

Dataset
The dataset contains the following features:

Dependents: Whether the customer has dependents (Yes/No).
Tenure: Duration of subscription in months.
OnlineSecurity: Whether the customer has online security (Yes/No/No internet service).
OnlineBackup: Whether the customer has online backup (Yes/No/No internet service).
InternetService: Type of internet service (DSL/Fiber optic/No).
DeviceProtection: Whether the customer has device protection (Yes/No/No internet service).
TechSupport: Whether the customer has tech support (Yes/No/No internet service).
Contract: Contract type (Month-to-month/One year/Two year).
PaperlessBilling: Whether the customer uses paperless billing (Yes/No).
MonthlyCharges: Monthly subscription cost.
Churn: Target variable indicating churn (Yes/No).

Methodology
Analytic Approach

Analyze historical customer data to identify patterns distinguishing loyal customers from those who churn.
Use machine learning classification models to predict churn risk for active and new customers.
Focus on minimizing false negatives, as losing customers is costlier than unnecessary retention efforts.

Data Preprocessing

Handle missing values using imputation techniques (e.g., KNNImputer, SimpleImputer).
Encode categorical variables using OneHotEncoder, OrdinalEncoder, or BinaryEncoder.
Scale numerical features (tenure, MonthlyCharges) using MinMaxScaler or StandardScaler.
Address class imbalance using techniques like NearMiss, SMOTE, or RandomOverSampler.

Modeling

Algorithms Tested: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, KNeighborsClassifier, GradientBoostingClassifier.
Best Model: Logistic Regression with NearMiss resampling and Grid Search hyperparameter tuning.
Feature Selection: Evaluated full features, correlation-based selection, and importance-based selection.
Evaluation Metric: F2 score (prioritizing recall to minimize false negatives) and recall.

Key Findings

Best Model Performance:
F2 score: 0.7412
Recall: 85.27%


Key Features:
Negative Impact on Churn: Longer tenure, longer contract duration, and online security/backup services reduce churn risk.
Positive Impact on Churn: Higher monthly charges and fiber optic internet service increase churn risk.


Full-feature models outperformed feature-selected models, indicating predictive information is distributed across many features.

Business Impact

Churn Rate: 26.5% (1286 out of 4853 customers churned).
Cost Assumptions:
Customer Acquisition Cost (CAC): $694 per new customer.
Retention Cost: $173.5 per customer (1/4 of CAC).


Without Model:
Cost to replace 258 churned customers: $179,052.


With Model:
Retention cost for 452 predicted churners: $78,442.
Acquisition cost for 38 missed churners: $26,372.
Total cost: $104,814.


Savings: $74,238 (41% reduction in churn-related costs).

Conclusions

The Logistic Regression model with NearMiss and Grid Search achieved the best performance, with an F2 score of 0.7412 and 85.27% recall.
Full-feature models outperformed feature-selected models, suggesting minor features contribute to predictive power.
NearMiss resampling balanced the dataset effectively without introducing noise, unlike ADASYN.
The model enables targeted retention, reducing churn costs by over 41%.

Recommendations
Modeling

Cohort Analysis: Evaluate churn patterns based on customer signup periods or activity.
Algorithm Exploration: Test additional models and systematically tune hyperparameters.
Model Monitoring: Regularly assess model performance to adapt to changing customer behavior.
Insight Deepening: Analyze model outputs to identify root causes of churn for targeted retention programs.

Business

Promote Long-Term Contracts: Offer incentives (discounts, exclusive features) to encourage longer contracts, reducing churn risk.
Review Pricing Structure: Adjust monthly charges to be competitive, as high costs drive churn.
Targeted Marketing: Focus campaigns on high-risk customer segments with personalized offers.
Enhance Onboarding: Provide attractive incentives and responsive support for new customers to boost long-term retention.

Installation
To run the project, install the required dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn jcopml category_encoders imblearn dython xgboost lightgbm

Usage

Clone the repository:git clone <repository-url>


Navigate to the project directory:cd telco-churn-prediction


Open the Jupyter notebook (Capstone_TELCO-2.ipynb) in Jupyter Notebook or Google Colab.
Run the cells sequentially to preprocess data, train models, and evaluate results.

License
This project is licensed under the MIT License.
Acknowledgments

Dataset provided by Telco for churn prediction.
Libraries: scikit-learn, jcopml, category_encoders, imblearn, dython, xgboost, lightgbm.
References: First Page Sage (CAC data), Forbes (retention cost insights).

