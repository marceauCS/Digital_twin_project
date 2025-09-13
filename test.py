import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import warnings
from scipy.signal import butter, filtfilt

from utility import read_all_csvs_one_test
from utility import run_cv_one_motor
from utility import read_all_test_data_from_path
from utility import read_all_test_data_from_path, show_reg_result,extract_selected_feature, prepare_sliding_window, FaultDetectReg

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import f_regression
from functions import *


# %% 
# ### Read data and pre_processing
utility_path = '../'
sys.path.insert(1, utility_path)


n_int = 20

# Subfunction for data preprocessing.
def pre_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    ### Description
    Preprocess a dataset by performing:
    - Outlier removal (position, temperature, voltage)
    - Low-pass filtering and smoothing
    - Feature engineering (difference between current and previous data points)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing columns:
        - 'position': Position data (expected range: 0-1000)
        - 'temperature': Temperature data (expected range: 0-100 °C)
        - 'voltage': Voltage data (expected range: 6000-8000)
    """

    # 1. Define Helper Functions
    def butter_lowpass(cutoff: float, fs: float, order: int = 5):
        """
        Design a Butterworth low-pass filter.

        Parameters
        ----------
        cutoff : float
            Cutoff frequency of the filter.
        fs : float
            Sampling frequency of the data.
        order : int, optional
            Order of the filter (default is 5).

        Returns
        -------
        tuple
            Filter coefficients (b, a).
        """
        nyquist = 0.5 * fs  # Nyquist frequency = half of sampling frequency
        normal_cutoff = cutoff / nyquist  # Normalize cutoff frequency
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def lowpass_filter(data: pd.Series, cutoff_freq: float, sampling_freq: float, order: int = 5):
        """
        Apply a Butterworth low-pass filter to a signal.

        Parameters
        ----------
        data : pd.Series
            Input data to filter.
        cutoff_freq : float
            Cutoff frequency of the filter.
        sampling_freq : float
            Sampling frequency of the data.
        order : int, optional
            Order of the filter (default is 5).

        Returns
        -------
        np.ndarray
            Filtered data as a NumPy array.
        """
        b, a = butter_lowpass(cutoff_freq, sampling_freq, order=order)
        return filtfilt(b, a, data)

    # 2. Filtering Parameters
    cutoff_frequency = 0.8   # Cutoff frequency for smoothing (adjustable)
    sampling_frequency = 10  # Sampling frequency (assumes evenly spaced time series)

    # 3. Outlier Removal & Smoothing
    def customized_outlier_removal(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers and smooth signals for position, temperature, and voltage.

        - Invalid values are replaced with NaN and forward-filled.
        - Low-pass filtering and rolling averages are applied for smoothing.
        - Large temperature jumps are treated as anomalies and corrected.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing position, temperature, and voltage columns.

        Returns
        -------
        pd.DataFrame
            Cleaned and smoothed dataframe.
        """
        # ----- POSITION -----
        # Remove invalid positions outside range [0, 1000]
        df['position'] = df['position'].where(df['position'] <= 1000, np.nan)
        df['position'] = df['position'].where(df['position'] >= 0, np.nan)
        df['position'] = df['position'].ffill()  # Replace NaNs with last valid value
        df['position'] = lowpass_filter(df['position'], cutoff_frequency, sampling_frequency)
        df['position'] = df['position'].rolling(window=20, min_periods=1).mean()
        df['position'] = df['position'].round()  # Round to nearest integer

        # ----- TEMPERATURE -----
        # Keep only valid temperature range [0, 100 °C]
        df['temperature'] = df['temperature'].where(df['temperature'] <= 100, np.nan)
        df['temperature'] = df['temperature'].where(df['temperature'] >= 0, np.nan)
        df['temperature'] = df['temperature'].rolling(window=20, min_periods=1).mean()

        # Detect and remove large temperature jumps (spikes)
        threshold = 5  # Max allowed difference between consecutive readings
        prev_tmp = df['temperature'].shift(1)  # Previous temperature
        temp_diff = np.abs(df['temperature'] - prev_tmp)
        df.loc[temp_diff > threshold, 'temperature'] = np.nan
        df['temperature'] = df['temperature'].ffill()

        # ----- VOLTAGE -----
        # Keep only valid voltage range [6000, 8000]
        df['voltage'] = df['voltage'].where(df['voltage'] >= 6000, np.nan)
        df['voltage'] = df['voltage'].where(df['voltage'] <= 8000, np.nan)
        df['voltage'] = df['voltage'].ffill()
        df['voltage'] = lowpass_filter(df['voltage'], cutoff_frequency, sampling_frequency)
        df['voltage'] = df['voltage'].rolling(window=5, min_periods=1).mean()

    # Apply custom outlier removal
    customized_outlier_removal(df)

# Ignore warnings.
warnings.filterwarnings('ignore')

base_dictionary = r'C:/Users/marce/Documents/GitHub/digital_twin_robot/projects/maintenance_industry_4_2024/dataset/training_data/'
df_train = read_all_test_data_from_path(base_dictionary, pre_processing, is_plot=False)

# %%
# Pre-train the model.
# Get all the normal data
normal_test_id = ['20240105_164214',
    '20240105_165300',
    '20240105_165972',
    '20240320_152031',
    '20240320_153841',
    '20240320_155664',
    '20240321_122650',
    '20240325_135213',
    '20240325_152902',
    '20240426_141190',
    '20240426_141532',
    '20240426_141602',
    '20240426_141726',
    '20240426_141938',
    '20240426_141980',
    '20240503_163963',
    '20240503_164435',
    '20240503_164675',
    '20240503_165189'
]

df_experiment = df_train[df_train['test_condition'].isin(normal_test_id)]

# %%
# ### Testing data.

# %%
# Read all the dataset. Change to your dictionary if needed.
base_dictionary = r'C:/Users/marce/Documents/GitHub/digital_twin_robot/projects/maintenance_industry_4_2024/dataset/testing_data/'
df_test = read_all_test_data_from_path(base_dictionary, pre_processing, is_plot=False)

# %%
df_experiment

# %%
# ## Training the regression model.

# %%
import warnings
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def run_all_motors_validation(motor_label: int, drop_list: list):
    """
    ### Description
    Train and validate multiple regression models for motor temperature prediction, 
    perform hyperparameter tuning, and detect abnormal motor behavior.

    This function:
    - Prepares features and response variables for a given motor.
    - Applies sliding window feature enrichment for time-series data.
    - Trains multiple regression models (Linear, Ridge, Lasso, ElasticNet, Decision Tree).
    - Performs hyperparameter tuning using cross-validation.
    - Runs fault detection on test data.
    - Returns model predictions and best hyperparameters.

    Parameters
    ----------
    motor_label : int
        The index or label of the motor for which validation is performed.
    drop_list : list
        List of column names to drop from the dataset before model training.

    Returns
    -------
    tuple
        model_predictions : dict
            Dictionary containing model predictions for each trained model.
            Keys are formatted as `y_pred_ModelName`.
        best_params : dict
            Dictionary containing the best hyperparameters for each model.
    """

    # 1. Feature Preparation
    # Remove unnecessary columns and keep relevant features for training.
    feature_list_all = df_experiment.drop(columns=drop_list).columns.tolist()
    x_tr_org = df_experiment.drop(columns=drop_list + label_columns)  # Features
    y_temp_tr_org = df_experiment[f"data_motor_{motor_label}_label"]  # Response variable (labels)

    # 2. Sliding Window Feature Enrichment
    # Create time-series enriched features for better temporal learning.
    window_size = 70              # Number of samples per window
    sample_step = 30              # Step size to slide the window
    prediction_lead_time = 5      # How far ahead to predict
    threshold = 0.9               # Fault detection decision threshold
    abnormal_limit = 3            # Number of abnormal readings before flagging a fault

    x_tr, y_temp_tr = prepare_sliding_window(
        df_x=x_tr_org,
        y=y_temp_tr_org,
        window_size=window_size,
        sample_step=sample_step,
        prediction_lead_time=prediction_lead_time,
        mdl_type='reg'
    )

    warnings.filterwarnings('ignore')  # Suppress warnings during training

    # 3. Model Initialization
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'ElasticNet Regression': ElasticNet(),
        'Decision Tree Regression': DecisionTreeRegressor()
    }

    # Define hyperparameter grids for tuning
    param_grids = {
        'Linear Regression': {},
        'Ridge Regression': {'alpha': [0.001, 0.01, 0.1, 1]},
        'Lasso Regression': {'alpha': [0.001, 0.01, 0.1, 1]},
        'ElasticNet Regression': {
            'alpha': [0.001, 0.01, 0.1, 1],
            'l1_ratio': [0.001, 0.01, 0.1, 1]
        },
        'Decision Tree Regression': {
            'max_depth': [2, 3, 4],
            'min_samples_split': [2, 3, 4]
        }
    }

    model_predictions = {}
    best_params = {}

    # 4. Model Training, Hyperparameter Tuning & Validation
    for model_name, model in models.items():

        # Build pipeline: normalization → model
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),  # Normalize features
            ('model', model)
        ])

        # Format parameter grid for use with pipeline
        param_grid = {f'model__{key}': value for key, value in param_grids[model_name].items()}

        # Perform grid search cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(x_tr, y_temp_tr)

        # Store best model and its parameters
        mdl = grid_search.best_estimator_
        best_params[model_name] = grid_search.best_params_

        # 5. Fault Detection & Model Evaluation
        # Initialize custom regression-based fault detector
        detector_reg = FaultDetectReg(
            reg_mdl=mdl,
            threshold=threshold,
            abnormal_limit=abnormal_limit,
            window_size=window_size,
            sample_step=sample_step,
            pred_lead_time=prediction_lead_time
        )

        # Prepare test data for evaluation
        x_test_org, y_temp_test_org = extract_selected_feature(
            df_data=df_test,
            feature_list=feature_list_all,
            motor_idx=motor_label,
            mdl_type='reg'
        )

        # Make predictions on test set
        y_pred, y_response_test_pred = detector_reg.predict(
            df_x_test=x_test_org,
            y_response_test=y_temp_test_org,
            complement_truncation=True
        )

        # Store predictions
        model_predictions[f'y_pred_{model_name.replace(" ", "_")}'] = y_pred

    return model_predictions, best_params


# %% 
# # Motor 1

# %% 
# Feature Selection

# %%
drop_list1_label1 = ['data_motor_2_voltage', 'data_motor_3_voltage', 'data_motor_4_voltage', 'data_motor_5_voltage', 'data_motor_6_voltage', 'data_motor_6_position']

drop_list2_label1 = []


# %%
#model_predictions, best_params1 = run_all_motors_validation(1, drop_list1_label1)
model_predictions, best_params1 = run_all_motors_validation(1, drop_list2_label1)
y_pred1_Linear_Regression = model_predictions['y_pred_Linear_Regression']
y_pred1_Ridge_Regression = model_predictions['y_pred_Ridge_Regression']
y_pred1_Lasso_Regression = model_predictions['y_pred_Lasso_Regression']
y_pred1_ElasticNet_Regression= model_predictions['y_pred_ElasticNet Regression']
y_pred1_Decision_Tree_Regression= model_predictions['y_pred_Decision_Tree_Regression']

# %%
best_params1

# %% 
# # Motor 2

# %% 
# Feature Selection

# %%
drop_list1_label2 = ['data_motor_2_voltage', 'data_motor_3_voltage', 'data_motor_4_voltage', 'data_motor_5_voltage', 'data_motor_6_voltage', 'data_motor_5_temperature']

drop_list2_label2 = []

# %%
#model_predictions, best_params2 = run_all_motors_validation('data_motor_2_label', drop_list1_label2)
model_predictions, best_params2 = run_all_motors_validation('data_motor_2_label', drop_list2_label2)

y_pred2_Linear_Regression = model_predictions['y_pred_Linear_Regression']
y_pred2_Ridge_Regression = model_predictions['y_pred_Ridge_Regression']
y_pred2_Lasso_Regression = model_predictions['y_pred_Lasso_Regression']
y_pred2_ElasticNet_Regression= model_predictions['y_pred_ElasticNet Regression']
y_pred2_Decision_Tree_Regression= model_predictions['y_pred_Decision_Tree_Regression']

# %%
best_params2

# %% 
# # Motor 3

# %% 
# Feature Selection
# 

# %%
drop_list1_label3 = ['data_motor_2_voltage', 'data_motor_3_voltage', 'data_motor_4_voltage', 'data_motor_5_voltage', 'data_motor_6_voltage']
drop_list2_label3 = []

# %%
#model_predictions, best_params3 = run_all_motors_validation('data_motor_3_label', drop_list1_label3)
model_predictions, best_params3 = run_all_motors_validation('data_motor_3_label', drop_list2_label3)
y_pred3_Linear_Regression = model_predictions['y_pred_Linear_Regression']
y_pred3_Ridge_Regression = model_predictions['y_pred_Ridge_Regression']
y_pred3_Lasso_Regression = model_predictions['y_pred_Lasso_Regression']
y_pred3_ElasticNet_Regression= model_predictions['y_pred_ElasticNet Regression']
y_pred3_Decision_Tree_Regression= model_predictions['y_pred_Decision_Tree_Regression']

# %%
best_params3

# %% 
# # Motor 4

# %% 
# Feature Selection

# %%
drop_list1_label4= ['data_motor_2_voltage', 'data_motor_3_voltage', 'data_motor_4_voltage', 'data_motor_5_voltage', 'data_motor_6_voltage', 'data_motor_5_temperature']
drop_list2_label4 = []

# %%
# model_predictions,best_params4 = run_all_motors_validation('data_motor_4_label', drop_list1_label4)
model_predictions,best_params4 = run_all_motors_validation('data_motor_4_label', drop_list2_label4)

y_pred4_Linear_Regression = model_predictions['y_pred_Linear_Regression']
y_pred4_Ridge_Regression = model_predictions['y_pred_Ridge_Regression']
y_pred4_Lasso_Regression = model_predictions['y_pred_Lasso_Regression']
y_pred4_ElasticNet_Regression= model_predictions['y_pred_ElasticNet Regression']
y_pred4_Decision_Tree_Regression= model_predictions['y_pred_Decision_Tree_Regression']

# %%
best_params4

# %% 
# # Motor 5

# %% 
# Feature Selection

# %%
drop_list1_label5 = ['data_motor_2_voltage', 'data_motor_3_voltage', 'data_motor_4_voltage', 'data_motor_5_voltage', 'data_motor_6_voltage']
drop_list2_label5 = []

# %%
# model_predictions,best_params5 = run_all_motors_validation('data_motor_5_label', drop_list1_label5)
model_predictions,best_params5 = run_all_motors_validation('data_motor_5_label', drop_list2_label5)

y_pred5_Linear_Regression = model_predictions['y_pred_Linear_Regression']
y_pred5_Ridge_Regression = model_predictions['y_pred_Ridge_Regression']
y_pred5_Lasso_Regression = model_predictions['y_pred_Lasso_Regression']
y_pred5_ElasticNet_Regression= model_predictions['y_pred_ElasticNet Regression']
y_pred5_Decision_Tree_Regression= model_predictions['y_pred_Decision_Tree_Regression']

# %%
best_params5

# %% 
# # Motor 6

# %% 
# Feature Selection

# %%
drop_list1_label6 = ['data_motor_2_voltage', 'data_motor_3_voltage', 'data_motor_4_voltage', 'data_motor_5_voltage', 'data_motor_6_voltage', 'data_motor_6_position']
drop_list2_label6 = []

# %%
# model_predictions,best_params6 = run_all_motors_validation('data_motor_6_label', drop_list1_label6)
model_predictions,best_params6 = run_all_motors_validation('data_motor_6_label', drop_list2_label6)

y_pred6_Linear_Regression = model_predictions['y_pred_Linear_Regression']
y_pred6_Ridge_Regression = model_predictions['y_pred_Ridge_Regression']
y_pred6_Lasso_Regression = model_predictions['y_pred_Lasso_Regression']
y_pred6_ElasticNet_Regression= model_predictions['y_pred_ElasticNet Regression']
y_pred6_Decision_Tree_Regression= model_predictions['y_pred_Decision_Tree_Regression']

# %%
best_params6

# %% 
# ## Create csv file for submit Prediction

# %% 
# Linear_Regression

# %%
data_Linear_Regression = {
    'idx': range(len(y_pred1_Linear_Regression)),
    'data_motor_1_label': y_pred1_Linear_Regression,
    'data_motor_2_label': y_pred2_Linear_Regression,
    'data_motor_3_label': y_pred3_Linear_Regression,
    'data_motor_4_label': y_pred4_Linear_Regression,
    'data_motor_5_label': y_pred5_Linear_Regression,
    'data_motor_6_label': y_pred6_Linear_Regression
}

df_Linear_Regression = pd.DataFrame(data_Linear_Regression)

df_Linear_Regression.to_csv('motor_predictions_Linear_Regression.csv', index=False)

# %% 
# Ridge_Regression

# %%
data_Ridge_Regression = {
    'idx': range(len(y_pred1_Ridge_Regression)),
    'data_motor_1_label': y_pred1_Ridge_Regression,
    'data_motor_2_label': y_pred2_Ridge_Regression,
    'data_motor_3_label': y_pred3_Ridge_Regression,
    'data_motor_4_label': y_pred4_Ridge_Regression,
    'data_motor_5_label': y_pred5_Ridge_Regression,
    'data_motor_6_label': y_pred6_Ridge_Regression
}

df_Ridge_Regression = pd.DataFrame(data_Ridge_Regression)

df_Ridge_Regression.to_csv('motor_predictions_Ridge_Regression.csv', index=False)

# %% 
# Lasso_Regression

# %%
data_Lasso_Regression = {
    'idx': range(len(y_pred1_Linear_Regression)),
    'data_motor_1_label': y_pred1_Lasso_Regression,
    'data_motor_2_label': y_pred2_Lasso_Regression,
    'data_motor_3_label': y_pred3_Lasso_Regression,
    'data_motor_4_label': y_pred4_Lasso_Regression,
    'data_motor_5_label': y_pred5_Lasso_Regression,
    'data_motor_6_label': y_pred6_Lasso_Regression
}

df_Lasso_Regression = pd.DataFrame(data_Lasso_Regression)

df_Lasso_Regression.to_csv('motor_predictions_Lasso_Regression.csv', index=False)

# %% 
# ElasticNet_Regression

# %%
data_ElasticNet_Regression = {
    'idx': range(len(y_pred1_ElasticNet_Regression)),
    'data_motor_1_label': y_pred1_ElasticNet_Regression,
    'data_motor_2_label': y_pred2_ElasticNet_Regression,
    'data_motor_3_label': y_pred3_ElasticNet_Regression,
    'data_motor_4_label': y_pred4_ElasticNet_Regression,
    'data_motor_5_label': y_pred5_ElasticNet_Regression,
    'data_motor_6_label': y_pred6_ElasticNet_Regression
}

df_ElasticNet_Regression = pd.DataFrame(data_ElasticNet_Regression)

df_ElasticNet_Regression.to_csv('motor_predictions_ElasticNet_Regression.csv', index=False)

# %% 
# Decision_Tree_Regression

# %%
data_Decision_Tree_Regression = {
    'idx': range(len(y_pred1_Decision_Tree_Regression)),
    'data_motor_1_label': y_pred1_Decision_Tree_Regression,
    'data_motor_2_label': y_pred2_Decision_Tree_Regression,
    'data_motor_3_label': y_pred3_Decision_Tree_Regression,
    'data_motor_4_label': y_pred4_Decision_Tree_Regression,
    'data_motor_5_label': y_pred5_Decision_Tree_Regression,
    'data_motor_6_label': y_pred6_Decision_Tree_Regression
}

df_Decision_Tree_Regression = pd.DataFrame(data_Decision_Tree_Regression)

df_Decision_Tree_Regression.to_csv('motor_predictions_Decision_Tree_Regression.csv', index=False)


