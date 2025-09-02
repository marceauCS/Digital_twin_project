# %% [markdown]
# # Kaggle Regression
# 
# 

# %% [markdown]
# ### Libraries

# %%

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


# %% [markdown]
# ### Read data and pre_processing

# %%
utility_path = '../'
sys.path.insert(1, utility_path)


n_int = 20

# Subfunction for data preprocessing.
def pre_processing(df: pd.DataFrame):
    ''' ### Description
    Preprocess the data:
    - remove outliers
    - add new features about the difference between the current and previous n data point.
    '''
    
    # Function to design a Butterworth low-pass filter
    def butter_lowpass(cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a


    # Function to apply the Butterworth low-pass filter
    def lowpass_filter(data, cutoff_freq, sampling_freq, order=5):
        b, a = butter_lowpass(cutoff_freq, sampling_freq, order=order)
        filtered_data = filtfilt(b, a, data)
        return filtered_data


    # Set parameters for the low-pass filter
    cutoff_frequency = .8  # Adjust as needed
    sampling_frequency = 10  # Assuming your data is evenly spaced in time


    def customized_outlier_removal(df: pd.DataFrame):
        ''' # Description
        Remove outliers from the dataframe based on defined valid ranges. 
        Define a valid range of temperature and voltage. 
        Use ffil function to replace the invalid measurement with the previous value.
        '''
        df['position'] = df['position'].where(df['position'] <= 1000, np.nan)
        df['position'] = df['position'].where(df['position'] >= 0, np.nan)
        df['position'] = df['position'].ffill()
        df['position'] = lowpass_filter(df['position'], cutoff_frequency, sampling_frequency)
        df['position'] = df['position'].rolling(window=20, min_periods=1).mean()
        df['position'] = df['position'].round()

        df['temperature'] = df['temperature'].where(df['temperature'] <= 100, np.nan)
        df['temperature'] = df['temperature'].where(df['temperature'] >= 0, np.nan)
        df['temperature'] = df['temperature'].rolling(window=20, min_periods=1).mean()

        # Make sure that the difference between the current and previous temperature cannot be too large.
        # Define your threshold
        threshold = 5
        # Shift the 'temperature' column by one row to get the previous temperature
        prev_tmp = df['temperature'].shift(1)
        # Calculate the absolute difference between current and previous temperature
        temp_diff = np.abs(df['temperature'] - prev_tmp)
        # Set the temperature to NaN where the difference is larger than the threshold
        df.loc[temp_diff > threshold, 'temperature'] = np.nan
        df['temperature'] = df['temperature'].ffill()

        df['voltage'] = df['voltage'].where(df['voltage'] >= 6000, np.nan)
        df['voltage'] = df['voltage'].where(df['voltage'] <= 8000, np.nan)
        df['voltage'] = df['voltage'].ffill()
        df['voltage'] = lowpass_filter(df['voltage'], cutoff_frequency, sampling_frequency)
        df['voltage'] = df['voltage'].rolling(window=5, min_periods=1).mean()  

    # Start processing.
    customized_outlier_removal(df)

# Ignore warnings.
warnings.filterwarnings('ignore')

base_dictionary = r'C:\Users\lucas\Documents\GitHub\digital_twin_robot\projects\maintenance_industry_4_2024\dataset/training_data/'
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

# %% [markdown]
# ### Testing data.

# %%
# Read all the dataset. Change to your dictionary if needed.
base_dictionary = r'C:\Users\lucas\Documents\GitHub\digital_twin_robot\projects\maintenance_industry_4_2024\dataset\testing_data/'
df_test = read_all_test_data_from_path(base_dictionary, pre_processing, is_plot=False)

# %%
df_experiment

# %% [markdown]
# ## Training the regression model.

# %%
def run_all_motors_validation(motor_label, drop_list):
    feature_list_all = df_experiment.drop(columns=drop_list).columns.tolist()
    # Prepare feature and response of the training dataset.
    #x_tr_org, y_temp_tr_org = extract_selected_feature(df_data=df_experiment, feature_list=feature_list_all, motor_idx=motor_label, mdl_type='reg')

    x_tr_org = df_experiment.drop(columns=drop_list+label_columns)
    y_temp_tr_org = df_experiment[f"data_motor_{motor_label}_label"]
    # Enrich the features based on the sliding window.
    window_size = 70
    sample_step = 30
    prediction_lead_time = 5 
    threshold = .9
    abnormal_limit = 3

    x_tr, y_temp_tr = prepare_sliding_window(df_x=x_tr_org, y=y_temp_tr_org, window_size=window_size, sample_step=sample_step, prediction_lead_time=prediction_lead_time, mdl_type='reg')
    
    warnings.filterwarnings('ignore')
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'ElasticNet Regression': ElasticNet(),
        'Decision Tree Regression': DecisionTreeRegressor()
    }
    
    # Define hyperparameter grids
    param_grids = {
    'Linear Regression': {},  
    'Ridge Regression': {'regressor__alpha': [0.001 , 0.01 , 0.1 , 1]},
    'Lasso Regression': {'regressor__alpha': [0.001 , 0.01 , 0.1 , 1]},
    'ElasticNet Regression': {'regressor__alpha': [0.001 , 0.01 , 0.1 , 1], 'regressor__l1_ratio': [0.001 , 0.01 , 0.1 , 1]}, 
    'Decision Tree Regression': {'regressor__max_depth': [2,3,4], 'regressor__min_samples_split': [2,3,4]}
    }
    
    model_predictions = {}
    best_params = {}
    
    # Perform cross-validation, hyperparameter tuning, and evaluation
    for model_name, model in models.items():
        
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()), # Step 1 : Normalization
            ('model', model)
        ])
        param_grid = {f'model__{key}': value for key, value in param_grids[model_name].items()}
        
        # Hyperparameter tuning
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
        #grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='f1')
        grid_search.fit(x_tr, y_temp_tr)
        mdl = grid_search.best_estimator_
        best_params[model_name] = grid_search.best_params_
        
        # Define the fault detector.
        detector_reg = FaultDetectReg(reg_mdl=mdl, threshold=threshold, abnormal_limit=abnormal_limit, window_size=window_size, sample_step=sample_step, pred_lead_time=prediction_lead_time)
        
        # Prepare the testing data.
        x_test_org, y_temp_test_org = extract_selected_feature(df_data=df_test, feature_list=feature_list_all, motor_idx=motor_label, mdl_type='reg')
        
        
        # Make predicition.
        y_pred, y_response_test_pred = detector_reg.predict(df_x_test=x_test_org, y_response_test=y_temp_test_org, complement_truncation=True)
        
        model_predictions[f'y_pred_{model_name.replace(" ", "_")}'] = y_pred 
    
    return model_predictions, best_params


# %% [markdown]
# # Motor 1

# %% [markdown]
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

# %% [markdown]
# # Motor 2

# %% [markdown]
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

# %% [markdown]
# # Motor 3

# %% [markdown]
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

# %% [markdown]
# # Motor 4

# %% [markdown]
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

# %% [markdown]
# # Motor 5

# %% [markdown]
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

# %% [markdown]
# # Motor 6

# %% [markdown]
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

# %% [markdown]
# ## Create csv file for submit Prediction

# %% [markdown]
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

# %% [markdown]
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

# %% [markdown]
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

# %% [markdown]
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

# %% [markdown]
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


