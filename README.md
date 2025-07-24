# ECE720T32-Using-ML-for-CTLE-Circuit-Generation

This is a group project for UWaterloo ECE720T32 (ML for Chip Design) where we are trying to use ML to design CTLE Circuits.

## Project Description

This project focuses on leveraging machine learning to predict the performance of Continuous-Time Linear Equalizer (CTLE) circuits. The primary goal is to create models that can accurately predict key performance metrics, such as attenuation levels and eye-diagram characteristics at various frequencies, based on the circuit's design parameters. By doing so, we aim to accelerate the circuit design process and explore the design space more efficiently.

## Code Explanations

### Data Handling (`handleDataset.py`, `handleDatasetv2.ipynb`)

-   **`combine_csv_files`**: This function reads all CSV files from a specified folder, concatenates them into a single pandas DataFrame, and saves the combined data to a new CSV file.
-   **`combine_all_rates_for_each_region`**: This function is more specific to the project's data structure. It finds sets of files corresponding to different data rates (7G, 14G, 28G, 56G), merges their unique columns, and then combines all the processed sets into a single output file.

### Modeling (`baseline.ipynb`, `baselineV2.ipynb`)

-   **Data Loading and Preprocessing**: The notebooks start by loading the `DataV2.csv` file. A custom `IQROutlierCapper` transformer is defined to handle outliers. The preprocessing pipeline also includes `SimpleImputer` for handling missing values, `RobustScaler` for scaling numerical features, and `OneHotEncoder` for categorical features.
-   **Model Training**: The notebooks train and evaluate several regression models:
    -   **XGBoost**: A gradient boosting model known for its performance.
    -   **LightGBM**: Another gradient boosting model that is often faster than XGBoost.
    -   **DNN (TensorFlow/Keras)**: A deep neural network with dense layers, dropout, and batch normalization.
-   **Hyperparameter Tuning**: `RandomizedSearchCV` is used to find the best hyperparameters for the XGBoost and LightGBM models.
-   **Evaluation**: The models are evaluated using a variety of metrics, including Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R-squared (R²), and Mean Absolute Percentage Error (MAPE). The `evaluate_model` and `print_results` functions are used to calculate and display these metrics.

### Visualization (`visualization.ipynb`)

-   This notebook uses `seaborn` and `matplotlib` to generate several types of plots for exploratory data analysis:
    -   **Histograms**: To show the distribution of individual features.
    -   **Box Plots**: To identify outliers and understand the spread of the data.
    -   **Scatter Plots**: To visualize the relationship between two variables.
    -   **Correlation Heatmap**: To show the correlation between all pairs of features and targets.

## Instructions to Run

1.  **Prerequisites**: Ensure you have Python 3 and Jupyter Notebook installed.
2.  **Install Dependencies**: The required libraries are listed in the first cell of the `baselineV2.ipynb` notebook. You can install them using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm tensorflow scikeras joblib
    ```
3.  **Data Preparation**:
    -   The `DataV2.csv` file is already provided.
    -   If you wish to regenerate the dataset, you can run the `handleDatasetv2.ipynb` notebook.
4.  **Running the Models**:
    -   Open and run the `baselineV2.ipynb` notebook in Jupyter. This will train the models and display the evaluation results.
5.  **Viewing Visualizations**:
    -   Open and run the `visualization.ipynb` notebook to see the exploratory data analysis.
