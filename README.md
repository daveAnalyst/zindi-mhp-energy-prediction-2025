# Zindi Micro-Hydropower Energy Load Prediction Challenge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Optional license badge -->

This repository contains the code and solution approach for the [Zindi Micro-Hydropower Energy Load Prediction Challenge](https://zindi.africa/competitions/micro-hydropower-energy-load-prediction-challenge). The goal is to accurately predict daily energy load generation (kWh) per user for off-grid communities in Kalam, Pakistan, using MHP sensor data and climate indicators.

**Author:** `<David / daveAnalyst>`
**Competition Dates:** April 4 - April 12, 2025
**Final Public Leaderboard Score:** `<Best LB Score, e.g., 12.8939>` (`submission_v12_featsel.csv`)
**Final Private Leaderboard Score:** `<Update After Competition Ends>`
**Best Local CV Score (Mean Realistic):** `<Your Best Realistic CV Score, e.g., 1.6503>`

## Project Objective

To develop a machine learning model that predicts the total daily energy consumption (kWh) for individual data users connected to Micro-Hydropower Plants (MHPs). The model leverages time-series data from MHPs (voltage, current, kWh) and corresponding climate data (temperature, precipitation, wind speed, etc.) to improve forecasting accuracy, optimize energy distribution, and enhance system reliability in off-grid settings.

## Dataset

The data was provided by Zindi and curated by the Center for Intelligent Systems and Networks Research (CISNR), University of Engineering and Technology Peshawar. It consists of:

1.  **`Data.csv`:** Time-series data recorded at 5-minute intervals for multiple consumer devices/users. Includes voltage, current, power factor, and kWh readings. User identifiers are in the 'Source' column.
2.  **`Kalam Climate Data.xlsx`:** Corresponding climate indicators (temperature, precipitation, wind components, etc.) for the region.
3.  **`SampleSubmission.csv`:** Defines the required submission format with `ID` (YYYY-MM-DD_user_identifier) and `kwh` columns.

*Note: Raw data files are not included in this repository due to size but should be placed in a `data/` subfolder after downloading from Zindi.*

## Methodology

The solution follows an iterative machine learning workflow primarily implemented in a Jupyter Notebook (`zindi_mhp_dev.ipynb`):

1.  **Environment Setup:** Uses Python with a virtual environment (managed locally with `venv`, executed primarily in Google Colab for performance). Key libraries include `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `optuna`, `openpyxl`.
2.  **Data Loading & Preprocessing:**
    *   Loads MHP `.csv` data (using `usecols` for efficiency) and climate `.xlsx` data.
    *   Handles timestamp parsing and standardizes column names (e.g., `timestamp`, `user_id`).
    *   Aggregates 5-minute MHP data to `daily_kwh` per `user_id` and `date`.
    *   Aggregates climate data to daily statistics (mean/sum), calculating `wind_speed` from U/V components.
    *   Merges the two aggregated datasets.
    *   Handles missing values using `ffill` and `bfill`.
3.  **Feature Engineering:**
    *   **Date Features:** Extracts standard calendar features (year, month, day, dayofweek, dayofyear, weekofyear, quarter, is_weekend).
    *   **Lag Features:** Creates features representing past `daily_kwh` values for each user (lags: 1, 2, 3, 7, 14, 28 days). This proved crucial for performance.
    *   **User ID Feature:** Treats `user_id` as a `category` dtype, allowing the model to learn user-specific patterns.
    *   **(Experimented/Discarded):** Rolling window kWh features and climate interaction features were tested but did not improve leaderboard generalization in the final configurations.
    *   **(Experimented/Discarded):** Log transformation of the target variable improved local CV but worsened leaderboard performance.
    *   **NaN Handling:** Fills NaNs introduced by lags (or other feature engineering steps) with 0.
4.  **Modeling:**
    *   **Algorithm:** LightGBM (`lgb.LGBMRegressor`).
    *   **Validation Strategy:** 5-Fold Time Series Cross-Validation (`sklearn.model_selection.TimeSeriesSplit`) is used for robust local evaluation. Folds with zero-only target values are identified and excluded from the primary performance metric (`Mean Realistic CV RMSE`).
    *   **Hyperparameter Tuning:** Optuna was used to systematically search for optimal hyperparameters, minimizing the `Mean Realistic CV RMSE`. Key tuned parameters include `learning_rate`, `num_leaves`, `max_depth`, regularization terms (`lambda_l1`, `lambda_l2`), and others.
    *   **Final Model:** After cross-validation and tuning, a single LightGBM model is trained on the *entire* prepared training dataset (`df_train_full`) using the best hyperparameters found.
5.  **Feature Selection (Key Improvement):**
    *   Feature importance analysis was performed on the tuned model (v7 config).
    *   The least important features (identified as `'quarter', 'is_weekend', 'year', 'dayofweek', 'month', 'daily_kwh_lag_3', 'daily_kwh_lag_7'`) were removed from the feature set.
    *   Retraining the model with the reduced feature set and optimized parameters resulted in the best leaderboard score (v12).
6.  **Prediction Pipeline:**
    *   A dedicated pipeline prepares the test data based on `SampleSubmission.csv`.
    *   It combines relevant historical data with the future test structure to enable accurate calculation of lag features for the test period.
    *   It applies the same date feature engineering and `user_id` categorization.
    *   It handles NaNs consistently with the training pipeline.
    *   Uses the `final_model` to generate predictions.
    *   Clips negative predictions to zero.
7.  **Submission:** Creates a `.csv` file in the specified format (`ID`, `kwh`).

## Repository Structure

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/daveAnalyst/zindi-mhp-energy-prediction-2025.git
    cd zindi-mhp-energy-prediction-2025
    ```
2.  **Set up Environment:**
    *   **Recommended:** Use Google Colab. Upload the notebook (`zindi_mhp_dev.ipynb`).
    *   **(Alternative - Local):** Create a Python virtual environment and install requirements:
        ```bash
        python -m venv zindi_mhp_venv
        source zindi_mhp_venv/bin/activate # or .\zindi_mhp_venv\Scripts\activate on Windows
        pip install -r requirements.txt
        ```
3.  **Download Data:** Obtain `Data.zip`, `Climate Data.zip`, and `SampleSubmission.csv` from the Zindi competition page and place the *extracted* contents (`Data.csv`, `Kalam Climate Data.xlsx`, `SampleSubmission.csv`) into the `data/` directory.
4.  **Configure Colab (If using):**
    *   Upload data files to a designated Google Drive folder.
    *   Run the setup cell (Cell 1A) in the notebook, replacing placeholders for PAT (if repo is private), Git user info, and Google Drive path. This cell handles Drive mounting, cloning (if needed), package installation, and copying/unzipping data to the Colab runtime.
5.  **Run the Notebook:** Execute the cells in `zindi_mhp_dev.ipynb` sequentially.
    *   Cell 8 defines features (ensure the desired version - e.g., with feature selection - is active).
    *   Cell 9 runs Time Series CV (note the realistic CV score).
    *   Cell 10 trains the final model using specified parameters.
    *   Cell 10A (optional) plots feature importance.
    *   Cell 16 prepares the test set and makes predictions.
    *   Cell 18 generates the submission file (e.g., `submission_v12_featsel.csv`).
6.  **Submit:** Upload the generated submission CSV to the Zindi competition page.

## Key Findings & Future Work

*   Lag features and User ID encoding were the most impactful feature engineering steps locally.
*   Systematic hyperparameter tuning (Optuna) significantly improved local CV scores.
*   **Feature selection** (removing weak features identified via importance analysis) was crucial for improving generalization and achieving the best leaderboard score (~12.89).
*   A significant gap between local CV performance (~1.65) and public LB performance (~12.89) persists, likely due to differences between the training data distribution/patterns and those in the specific public test set slice.
*   Target transformation (log) and climate interaction features did not improve leaderboard scores in these experiments.

**Potential Future Directions:**

*   Deeper analysis of train vs. test data distributions and time periods.
*   Explore different NaN filling strategies.
*   Experiment with different model types (XGBoost, CatBoost, potentially time-series specific models like N-BEATS if data structure allows).
*   More sophisticated feature engineering related to grid events or specific user types (if identifiable).
*   Ensembling techniques.

---
