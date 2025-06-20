📊 Employee Attrition Prediction with Deep Learning
Predicting employee turnover using machine learning, deep neural networks, and model interpretability techniques like SHAP.
🚀 Project Overview
This project analyzes employee behavioral and performance data to predict attrition at Salifort Motors, a fictional company. By leveraging both traditional ML models and a tuned deep neural network, the project achieves high predictive performance while maintaining interpretability and actionable insights for HR stakeholders.

📁 Dataset
The dataset contains employee-level information from departments like Sales, IT, and Support. It includes:

Feature	Description
satisfaction_level	Employee satisfaction score (0–1)
last_evaluation	Last performance evaluation score (0–1)
number_project	Total number of projects handled
average_montly_hours	Average monthly working hours
time_spend_company	Tenure with company (in years)
Work_accident	Whether the employee had a work accident
left	Target: 1 = employee left, 0 = stayed
promotion_last_5years	Was promoted in the last 5 years (1/0)
Department	Department/Functional group
salary	Salary level (low, medium, high)
🧪 Exploratory Data Analysis
High attrition among low-salary, unpromoted, and overworked employees.

Strong inverse relationship between satisfaction level and attrition.

Tenure spikes (especially Year 3–4) coincide with resignation likelihood.

Visuals include:

Heatmaps

Boxplots

Correlation matrix

Pairwise scatterplots

Department-wise attrition rates

🔧 Feature Engineering
Created several derived features:

tenure_category: Categorical bins for tenure years (Junior/Mid/Senior)

workload_ratio: average_montly_hours / number_project

performance_change: Delta between last_evaluation and satisfaction_level

Categorical encoding via Label Encoding and One-Hot Encoding. Features scaled using StandardScaler.

🤖 Machine Learning Models
Baselines
Logistic Regression: 84% accuracy

Random Forest: 96% accuracy with strong feature importance

XGBoost: ~96.7% accuracy (best-performing classic model)

Deep Learning (Keras + TensorFlow)
Two hidden layers with ReLU activations and Dropout

Tuned via Keras Tuner (RandomSearch)

Final architecture:

Dense(96) → Dropout(0.3) → Dense(96) → Dropout(0.2) → Dense(1)

Validation Accuracy: ~96.8%

EarlyStopping used to prevent overfitting

📉 Model Evaluation
Evaluated using:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1)

ROC Curves for comparison across models

🔍 Model Explainability with SHAP
Used SHAP to explain:

Global feature importance via beeswarm plots

Local prediction reasoning via waterfall plots

Most influential features:

satisfaction_level

time_spend_company

number_project

salary

💡 Key Insights & Recommendations
Finding	Action Item
Low satisfaction, high work hours	Introduce well-being & workload balancing
Peak attrition at 3–4 years	Offer career development around those years
Promotions impact retention	Establish fair, transparent advancement paths
Low salary = higher churn	Conduct competitive compensation benchmarking
Sales & support: higher attrition	Role-specific engagement strategies
🧰 Tools & Tech
Python 3.9+

Pandas, NumPy, Scikit-learn

TensorFlow & Keras

Keras Tuner (for hyperparameter tuning)

SHAP (for model explainability)

Matplotlib, Seaborn