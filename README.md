# Credit Card Fraud Detection

This project performs an in-depth exploratory data analysis (EDA) on a credit card transaction dataset and builds several machine learning models to detect fraudulent transactions. The primary challenge is the highly imbalanced nature of the dataset, where fraudulent transactions account for only **0.172%** of the total.

The analysis is contained in the `Credit_Card_Fraud_Detection.ipynb` notebook.

## Key Findings from EDA

1.  **Data Characteristics**:
    *   The dataset is clean with no missing values.
    *   Features `V1` through `V28` are anonymized PCA components. `Time` and `Amount` are the only non-transformed features.

2.  **Time Analysis - "Golden Hours" for Fraud**:
    *   Transactions follow a clear daily (diurnal) pattern, with a sharp dip in volume during the early morning (European time).
    *   The fraud rate spikes dramatically at **2 AM**, when transaction volume is lowest. The fraud rate at this time is **9.9 times higher** than the daily average, suggesting fraudsters exploit periods of low activity.
    *   Fraudulent transactions often occur in rapid **"bursts"**, with multiple fraudulent events happening in quick succession.

3.  **Amount Analysis - The Mean-Median Paradox**:
    *   Fraudulent transactions have a higher mean amount (\$122.21) than normal ones (\$88.29), but a much lower median amount (\$9.25 vs \$22.00).
    *   This indicates that while most fraudulent transactions are for small amounts, a few are for extremely large values, skewing the mean.
    *   **73.6% of all fraudulent transactions are for amounts under \$100.**

4.  **Feature Analysis & Visualization**:
    *   Several PCA features, especially **V17, V14, V12, V10, V4, and V11**, show significantly different distributions for fraudulent vs. normal transactions, making them strong predictors.
    *   Using UMAP for dimensionality reduction, fraudulent transactions form **distinct, separable clusters** away from the main cluster of normal transactions. This confirms that fraud is not random noise and is highly detectable by ML algorithms.

## Feature Engineering

Based on the EDA, the following features were engineered:
*   **Time Transformation**: `Time` was converted from seconds to cyclical `Time_sin` and `Time_cos` features to capture the 24-hour periodicity.
*   **Amount Scaling**: `Amount` was log-transformed (`log1p`) and scaled using `RobustScaler` to handle its extreme skewness and sensitivity to outliers.

## Modeling and Results

Four classification models were trained using the engineered features. The dataset was split 80/20, and **SMOTE** was applied to the training set to handle class imbalance.

| Model                | AUPRC | AUROC | Precision | Recall |
| -------------------- | ----- | ----- | --------- | ------ |
| **XGBoost**          | 0.853 | 0.960 | **0.922** | 0.847  |
| **LightGBM**         | 0.846 | 0.965 | 0.912     | 0.847  |
| **RandomForest**     | 0.822 | 0.971 | 0.792     | 0.857  |
| **LogisticRegression** | 0.748 | 0.962 | 0.556     | 0.867  |

**Conclusion**: XGBoost and LightGBM performed the best, delivering a strong balance between high precision and high recall, as measured by the AUPRC score which is the most relevant metric for this imbalanced problem.

## How to Run

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2.  Navigate to the project directory:
    ```bash
    cd credit-card-fraud-detection
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Launch Jupyter Notebook and open `Credit_Card_Fraud_Detection.ipynb`:
    ```bash
    jupyter notebook
    ```