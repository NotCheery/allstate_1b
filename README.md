#  Predicting Auto Claims Severity

### 👥 **Team Members**

| Name             | GitHub Handle | Contribution                                   |
|------------------|---------------|------------------------------------------------|
| Tahia Islam      | @NotCheery    | Ridge Regression Model                         |
| Mihn Le          | @lqminhhh     | Generalized Linear Regression Model            |
| Samikha Srinivasan| @SamikhaS-rgb | Random Forest Model and Project Management    |
| Sanchita Agrawal     | @sanchita362      | XGBoost Model                              |
| Shemarie Irons       | @shemarieirons    | Lasso Regression model                     |
| Dieu Anh Trinh     | @audreydieuanh   | LightGBM Model                               |
| Ahmed Alamin      | @ahmedalaminn   | Neural Networks with Tensorflow                  |
---

## 🎯 **Project Highlights**
- Developed various predictive models to address the challenge of forecasting high-variance insurance claim costs.
- Achieved a 41.5% reduction in Mean Absolute Error (MAE) for our best model (LightGBM), decreasing prediction error from a baseline of $1,970.11 to $1,151.55
- Generated actionable business strategies, including automated claims triage workflows and fraud alert systems, to optimize operational efficiency and resource planning.

---

## 👩🏽‍💻 **Setup and Installation**

1. Clone the Repository

    Open your VSCode terminal and run the following command to clone the repository to your local machine:

        git clone https://github.com/NotCheery/allstate_1b.git

        cd allstate_1b

2. Install Dependencies

    Ensure you have Python installed. You can install the required libraries using pip

        pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm tensorflow jupyter

3. Set Up the Data

    **Note: The dataset is not included in this repository due to size constraints.**

    Download the datasets (claims_data.csv and data_dictionary.xlsx) from the Google Drive provided by the Challenge Advisor.

    Create a folder named data in the root directory of the project.

    Place the downloaded files inside the data/ folder.

4. Run the Analysis
    Launch Jupyter Notebook to view the project:
            
        jupyter notebook

---

## 🏗️ **Project Overview**

This project was developed as part of the **Break Through Tech AI Program**, an initiative designed to bridge the gap between academic theory and real-world AI application. Working within the program's "AI Studio," our team collaborated directly with our host company, Allstate Insurance Company.

The primary objective was to address the challenge of predicting insurance claim severity. The scope included performing rigorous Exploratory Data Analysis (EDA), handling data skewness through log-transformations, and engineering a robust machine learning pipeline. 

**Real-World Impact**: Accurately predicting claim costs is critical for the insurance industry. A precise model serves as an automated system allowing Allstate to fast-track low-cost claims for immediate payment (improving customer satisfaction) while flagging high-cost or anomalous claims for senior review (improving fraud detection and financial reserving). Our solution aims to optimize this operational workflow, potentially saving thousands of hours in manual review time.

---

## 📊 **Data Exploration**

**The dataset(s) used:**
- claims_data.csv and data_dictionary.xlsx provided by Allstate Challenge Advisor
- We used claims_data.csv and loaded into our notebooks as a DataFrame
- claims_data.csv size: (188318, 132)
- claims_data.csv type: Numerical and Categorical

**Data Exploration and Preprocessing**
- Target (loss) was right-skewed, hence we applied log transformation
- We explored features using heatmap, correlation matrix, boxplots, etc.
- The exploratory data analysis helped us idenitfy pre-processing steps to feed a clean dataset to train and test our models.
- Some challenges we faced when working with these datasets was understanding what the data and features mean (since it's anonymized).

![alt text](target.png)
---

## 🧠 **Model Development**

1. **Training Setup**

    To ensure robust evaluation and prevent data leakage, we implemented a strict validation framework:

    - **Data Split**: The dataset was split into 80% training and 20% testing sets (test_size=0.2, random_state=42).

    - **Target Transformation**: The target variable loss was log-transformed (np.log1p) during training to address the heavy right-skew inherent in insurance claim data. Predictions were converted back to the original dollar scale (np.expm1) for final evaluation.

2. **Models Implemented**
    
    We explored a diverse range of algorithms to balance interpretability with predictive power:

    - **Linear Baselines**: Ridge Regression, Lasso Regression, Elastic Net, General Linear Model (GLM).

    - **Tree-Based Ensembles**: Random Forest, XGBoost, LightGBM.

    - **Deep Learning**: TensorFlow (Neural Network).

3. **Optimization Strategies**
    
    - **Hyperparameter Tuning:**
        - Linear Models: Utilized GridSearchCV to rigorously test and identify the optimal alpha (penalty strength) for Ridge and Lasso, and the best l1_ratio for Elastic Net.

        - Tree Models: Optimized learning rates, depth, and estimators for LightGBM and XGBoost to prevent overfitting.

    - **Feature Selection:**

        - XGBoost: Analyzed feature importance to identify the Top 20 predictors, creating a reduced dataset that improved training speed and generalization.

        - LightGBM: Filtered for the Top 30 features while tuning hyperparameters to maximize the model's efficiency on the high-dimensional data.


---

## 📈 **Results & Key Findings**

Here's a table of each model and its performance metrics

### 🏆 Model Performance Summary

| Model | MAE (Original Scale) |
| :--- | :--- |
| **LightGBM** | **$1,151.41** |
| **XGBoost** | $1,170.04 |
| **Random Forest** | $1,192.80 |
| **TensorFlow** | $1,224.15 |
| **Ridge Regression** | $1,243.45 |
| **Lasso Regression** | $1,311.73 |
| **Elastic Net** | $1,243.09|

**Insight:** All models outperform the baseline mean MAE of 1970.11, with tree-based and deep learning models capturing nonlinear relationships more effectively than linear methods. We determined the best performing model, **LightGBM**, by Mean Absolute Error. 

---

## 🚀 **Next Steps**

While our current models deliver strong predictive performance, there are several key areas for future improvement and operational integration:

- **Incorporating Unstructured Data**: Our current models rely solely on structured tabular data. A major opportunity lies in using NLP (Natural Language Processing) to analyze claim notes or Computer Vision to assess accident photos. Merging this "multimodal" data could significantly boost accuracy for complex claims.

- **Addressing Model Interpretability**: The best-performing model (LightGBM) is a "black box," making it harder to explain decisions to customers. Future work would involve implementing SHAP (SHapley Additive exPlanations) values to provide real-time, explainable reasons for every price prediction.

- **Drift Monitoring**: As claim behaviors change over time (e.g., inflation, new car tech), the model's accuracy will degrade. We plan to build a Model Monitoring Dashboard to track performance drift and trigger automatic retraining when data patterns shift.

- **Real-Time Deployment**: Moving the model from a notebook to a REST API would allow claims adjusters to get instant pricing estimates the moment a First Notice of Loss (FNOL) is filed.

---

## 📝 **License**
This project is licensed under the MIT License.

---

## 🙏 **Acknowledgements** 

We would like to thank our coach, Eric Bayless and our Challenge Advisors, Krystal Smuda and Sedo Senou for their mentorship and support. Thank you to everyone at Break Through Tech who gave us this amazing opportunity!

