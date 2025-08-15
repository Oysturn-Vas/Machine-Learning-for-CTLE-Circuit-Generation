# ECE720T32 - Using Machine Learning for CTLE Circuit Generation

## 👥 Project Team

| Name                  | Email             |
| --------------------- | ----------------- |
| **Yufeng (Bob) Zhou** | zhou@gmail.com    |
| **Oysturn Vas**       | ovas@uwaterloo.ca |
| **Saad Imran Rana**   | saad@gmail.com    |

_University of Waterloo - ECE720T32 (ML for Chip Design)_

## Project Description

This project focuses on leveraging machine learning to predict the performance of Continuous-Time Linear Equalizer (CTLE) circuits. The primary goal is to create models that can accurately predict key performance metrics, such as attenuation levels and eye-diagram characteristics at various frequencies, based on the circuit's design parameters. By doing so, we aim to accelerate the circuit design process and explore the design space more efficiently.

This is a group project for **UWaterloo ECE720T32 (ML for Chip Design)** where we explore the application of various machine learning techniques to electronic circuit design optimization.

## 🎯 Project Objectives

1. **Forward Modeling**: Predict CTLE circuit performance metrics from design parameters
2. **Inverse Optimization**: Generate optimal circuit parameters for desired performance targets
3. **Design Space Exploration**: Understand the relationship between circuit parameters and performance
4. **Performance Analysis**: Compare different ML approaches for circuit modeling

## 📊 Dataset Description

The project uses two main datasets:

### Data.csv (Version 1)

- **Size**: 1000 samples × 22 features
- **Features**: Circuit parameters (fW, current, ind, Rd, Cs, Rs, Stage 1 Region)
- **Targets**: Channel and stage attenuations at different frequencies (0.1G, 3.5G, 7G, 14G, 28G)

### DataV2.csv (Version 2) - Primary Dataset

- **Size**: 10,000 samples × 39 features
- **Enhanced Features**: Same circuit parameters plus additional measurements
- **Extended Targets**:
  - Attenuation measurements at multiple frequencies
  - Eye diagram characteristics (height and width) at 7G, 14G, 28G, and 56G
  - Hard constraints and region classifications

#### Key Features:

- **Circuit Parameters**: `fW`, `current`, `ind`, `Rd`, `Cs`, `Rs`
- **Channel Attenuations**: At 0.1G, 3.5G, 7G, 14G, 28G frequencies
- **Stage Attenuations**: Individual stage performance metrics
- **Eye Diagram Metrics**: Height and width measurements for signal quality assessment
- **Constraints**: Hard constraints and region classifications

## 🏗️ Project Structure

```
├── Data.csv                           # Original dataset (V1)
├── DataV2.csv                        # Enhanced dataset (V2) - 10K samples
├── requirements.txt                   # Python dependencies
├── LICENSE                           # Project license
│
├── 📓 Notebooks/
│   ├── ctle_models_V1.ipynb         # Version 1 modeling pipeline
│   ├── ctle_models_V2.ipynb         # Version 2 enhanced pipeline
│   ├── preprocess_dataset.ipynb     # Data preprocessing utilities
│   └── visualization.ipynb          # Exploratory data analysis
│
├── 📁 Dataset/                       # Raw dataset folder (V1)
├── 📁 Dataset_v2/                    # Raw dataset folder (V2)
│
├── 📁 ctle_circuit_generation_with_DNN_models_V1/
│   ├── *.png                        # V1 visualization outputs
│   ├── *.joblib                     # V1 trained models
│   ├── *.csv                        # V1 results
│   └── output_file                  # V1 processing outputs
│
└── 📁 ctle_circuit_generation_with_DNN_models_V2/
    ├── *.png                        # V2 visualization outputs
    ├── *.joblib                     # V2 trained models
    ├── *.csv                        # V2 results
    └── metadata.json                # V2 comprehensive metadata
```

## 🔧 Technical Implementation

### Machine Learning Pipeline

#### 1. Data Preprocessing

- **Custom IQR Outlier Capper**: Handles outliers using Interquartile Range method
- **Missing Value Imputation**: SimpleImputer for handling missing data
- **Feature Scaling**: RobustScaler for numerical features
- **Categorical Encoding**: OneHotEncoder for categorical variables

```python
class IQROutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_bound_mult=1.5, upper_bound_mult=1.5):
        # Custom outlier capping using IQR method
```

#### 2. Model Architecture

**Version 1 Models:**

- XGBoost Regressor
- LightGBM Regressor
- Multi-Layer Perceptron (MLP)

**Version 2 Enhanced Models:**

- XGBoost Robust (Chained)
- LightGBM Robust (Chained)
- Neural Network Robust (Chained)
- Random Forest Robust
- Gradient Boosting Robust

#### 3. Target Classification System (V2)

The V2 system intelligently categorizes targets based on data quality:

- **Good Targets** (12): Stage attenuations and eye heights with sufficient data
- **Sparse Targets** (4): Eye heights at higher frequencies with limited data
- **Constant Targets** (8): Eye widths with minimal variation
- **Problematic Targets**: Targets with data quality issues

### Performance Metrics

#### Model Evaluation

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (R-squared Score)
- **MAPE** (Mean Absolute Percentage Error)

#### Best Model Performance (V2 - XGBoost Robust)

- **Overall R²**: 0.612
- **Overall RMSE**: 0.230
- **Overall MAE**: 0.075
- **Training Time**: 4.54 seconds

#### Individual Target Performance Categories:

- **Excellent** (R² > 0.9): 12 targets - Stage attenuations and primary eye heights
- **Good** (R² 0.8-0.9): 2 targets - Higher frequency eye heights
- **Fair** (R² 0.5-0.8): 2 targets - 56G eye heights
- **Negative** (R² < 0): 8 targets - Eye width measurements (constant values)

## 🎨 Visualization and Analysis

### Comprehensive Visualization Suite

1. **Data Overview**: Dataset statistics and distribution analysis
2. **Feature Distributions**: Histograms and box plots for all features
3. **Target Correlations**: Correlation matrices and heatmaps
4. **Preprocessing Effects**: Before/after preprocessing comparisons
5. **Model Comparison**: Performance metrics across all models
6. **Per-Target Performance**: Individual target prediction accuracy
7. **Feature Importance**: Tree-based model feature rankings
8. **Prediction Analysis**: Actual vs predicted scatter plots
9. **Inverse Optimization**: Parameter optimization results
10. **Optimized Parameters**: Generated circuit parameter distributions

### Key Insights from Analysis

#### Version 1 (1000 samples):

- **Perfect predictions** achieved due to dataset size limitations
- **MLP superiority** on extremely small datasets
- **Overfitting concerns** with near-perfect R² scores
- **Limited practical applicability** due to generalization issues

#### Version 2 (10,000 samples):

- **Stage attenuations** are highly predictable (R² > 0.99)
- **Eye diagram heights** show good predictability at lower frequencies
- **Eye diagram widths** remain constant across the dataset
- **Higher frequency predictions** (56G) are more challenging
- **XGBoost and LightGBM** consistently outperform neural networks
- **Realistic performance metrics** suitable for production deployment

## 🚀 Advanced Features

### Inverse Optimization System

The project includes a sophisticated inverse optimization system that:

1. **Target Setting**: Define desired performance characteristics
2. **Parameter Optimization**: Use optimization algorithms to find optimal circuit parameters
3. **Constraint Handling**: Respect physical and design constraints
4. **Multi-Objective Optimization**: Balance multiple performance metrics
5. **Validation**: Verify optimized parameters produce desired results

### Robust Model Training

- **Cross-Validation**: K-fold validation for reliable performance estimates
- **Hyperparameter Tuning**: RandomizedSearchCV for optimal parameters
- **Model Persistence**: Joblib serialization for model deployment
- **Comprehensive Logging**: Detailed training and evaluation logs

## 📋 Requirements and Installation

### Prerequisites

- Python 3.7+
- Jupyter Notebook/Lab

### Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm tensorflow scikeras joblib shap
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Usage Instructions

### 1. Data Preparation

```bash
# The DataV2.csv file is ready to use
# For custom datasets, run:
jupyter notebook preprocess_dataset.ipynb
```

### 2. Exploratory Data Analysis

```bash
jupyter notebook visualization.ipynb
```

### 3. Model Training and Evaluation

**Version 1 (Basic Pipeline):**

```bash
jupyter notebook ctle_models_V1.ipynb
```

**Version 2 (Enhanced Pipeline - Recommended):**

```bash
jupyter notebook ctle_models_V2.ipynb
```

### 4. Results Analysis

After running the notebooks, check the output directories:

**Version 1 Outputs:**

- **Models**: `ctle_circuit_generation_with_DNN_models_V1/*.joblib`
- **Visualizations**: `ctle_circuit_generation_with_DNN_models_V1/*.png`
- **Results**: `ctle_circuit_generation_with_DNN_models_V1/*.csv`
- **Optimization Results**: `ctle_circuit_generation_with_DNN_models_V1/inverse_optimization_results_fixed.csv`

**Version 2 Outputs (Recommended):**

- **Models**: `ctle_circuit_generation_with_DNN_models_V2/*.joblib`
- **Visualizations**: `ctle_circuit_generation_with_DNN_models_V2/*.png`
- **Results**: `ctle_circuit_generation_with_DNN_models_V2/*.csv`
- **Metadata**: `ctle_circuit_generation_with_DNN_models_V2/metadata.json`

## 📈 Results Summary

### Version Comparison Overview

The project evolved significantly from V1 to V2, with dramatic improvements in dataset size and model sophistication:

| Aspect                 | Version 1    | Version 2      |
| ---------------------- | ------------ | -------------- |
| **Dataset Size**       | 1000 samples | 10,000 samples |
| **Features**           | 22 features  | 39 features    |
| **Target Variables**   | 10 targets   | 24 targets     |
| **Best Model**         | MLP          | XGBoost Robust |
| **Best R²**            | **0.997**    | 0.612          |
| **Optimization Tests** | 20 tests     | 50+ tests      |

### Model Performance Comparison

#### Version 1 Results (Small Dataset - 1000 samples)

| Model    | Overall R² | RMSE      | MAE       | Performance Notes                   |
| -------- | ---------- | --------- | --------- | ----------------------------------- |
| **MLP**  | **0.997**  | **0.898** | **0.489** | **Best performer on small dataset** |
| XGBoost  | 0.989      | 1.913     | 1.326     | Strong but overfitted               |
| LightGBM | 0.987      | 2.040     | 1.427     | Consistent performance              |

#### Version 2 Results (Large Dataset - 10,000 samples)

| Model              | Overall R² | RMSE      | MAE       | Training Time (s) |
| ------------------ | ---------- | --------- | --------- | ----------------- |
| **XGBoost Robust** | **0.612**  | **0.230** | **0.075** | **4.54**          |
| LightGBM Robust    | 0.611      | 0.233     | 0.079     | 2.91              |
| Neural Network     | -212.1     | 0.227     | 0.067     | 104.52            |

### Performance Analysis by Version

#### Version 1 Characteristics:

- **Extremely high R² scores** (0.987-0.997) due to small dataset size
- **Potential overfitting** with only 1000 training samples
- **MLP dominance** - Neural networks performed best on limited data
- **Limited generalization** capability due to dataset constraints
- **Perfect optimization success** - All 20 inverse optimization tests succeeded

#### Version 2 Characteristics:

- **More realistic R² scores** (0.612) reflecting true model generalization
- **Robust evaluation** with 10,000 samples providing reliable metrics
- **Tree-based model superiority** - XGBoost and LightGBM outperform neural networks
- **Target-specific performance** - Excellent for stage attenuations (R² > 0.99), challenging for eye characteristics
- **Comprehensive optimization** - 50+ tests with detailed success analysis

### Inverse Optimization Results

#### Version 1 Optimization:

- **Success Rate**: 95% (19/20 tests successful)
- **Mean MSE Error**: ~190.36 (for successful optimizations)
- **Parameter Range**: Successfully optimized across full parameter space
- **Convergence**: Fast convergence due to simple model relationships

#### Version 2 Optimization:

- **Success Rate**: Variable (depends on target complexity)
- **Advanced Multi-Objective**: Balances attenuation and eye diagram targets
- **Constraint Handling**: Sophisticated constraint management
- **Realistic Performance**: Reflects real-world optimization challenges

### Key Findings

#### Dataset Size Impact:

1. **V1 (1000 samples)**: Achieved unrealistically high performance due to overfitting
2. **V2 (10,000 samples)**: Provides realistic, generalizable performance metrics
3. **Model Selection**: Neural networks excel on tiny datasets, tree-based models on larger datasets

#### Target Complexity:

1. **Stage attenuations**: Highly predictable across both versions (R² > 0.98)
2. **Eye diagram characteristics**: More challenging, especially at higher frequencies
3. **Frequency dependency**: Performance degrades at higher frequencies (28G, 56G)

#### Practical Implications:

1. **V1**: Proof of concept with limited practical applicability
2. **V2**: Production-ready system with realistic performance expectations
3. **Optimization**: V2 provides more robust and practical circuit parameter generation

### Model Evolution Insights

The dramatic difference between V1 and V2 results illustrates important machine learning principles:

- **Small Dataset Trap**: V1's exceptional performance (R² = 0.997) was misleading due to overfitting
- **Generalization Reality**: V2's moderate performance (R² = 0.612) represents true model capability
- **Model Selection**: Algorithm choice depends heavily on dataset characteristics
- **Evaluation Rigor**: Large datasets provide more reliable performance assessment

## 🤝 Contributing

This is an academic project for ECE720T32. For questions or collaboration:

1. Review the existing notebooks and documentation
2. Check the issues and discussions
3. Follow the established code structure and documentation standards

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🙏 Acknowledgments

- **University of Waterloo ECE720T32** course staff and instructors
- **CTLE Circuit Design** research community
- **Open-source ML libraries** that made this project possible

---
