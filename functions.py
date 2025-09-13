def smooth_data_moving_average(data, window_size):
    """
    Smooth a time series or dataframe using a simple moving average.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Input data to be smoothed.
    window_size : int
        Size of the moving window.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Smoothed data where each value is the mean of the current
        and previous `window_size - 1` values.
    """
    return data.rolling(window=window_size, min_periods=1).mean()


# List of label columns that might exist in the dataset
label_columns = [
    'data_motor_1_label', 'data_motor_2_label', 'data_motor_3_label',
    'data_motor_4_label', 'data_motor_5_label', 'data_motor_6_label'
]


def remove_outliers(data, alpha):
    """
    Remove outliers from a DataFrame using the Interquartile Range (IQR) method.

    Outliers are defined as points outside [Q1 - alpha*IQR, Q3 + alpha*IQR].
    If label columns are present, they are temporarily excluded from the calculation
    and reattached to the final cleaned DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset.
    alpha : float
        Scaling factor for the IQR range. A typical value is 1.5.

    Returns
    -------
    pandas.DataFrame
        DataFrame with rows containing outliers removed.
    """
    # Exclude label columns (if present) before computing IQR
    if all(col in data.columns for col in label_columns):
        data_without_labels = data.drop(columns=label_columns)
    else:
        data_without_labels = data.copy(deep=True)

    # Compute IQR bounds for each numeric column
    Q1 = data_without_labels.quantile(0.25, numeric_only=True)
    Q3 = data_without_labels.quantile(0.75, numeric_only=True)
    IQR = Q3 - Q1
    lower_bound = Q1 - alpha * IQR
    upper_bound = Q3 + alpha * IQR

    # Create mask for rows within acceptable range
    mask = ~(
        (data_without_labels.lt(lower_bound, axis=1)) |
        (data_without_labels.gt(upper_bound, axis=1))
    ).any(axis=1)

    return data[mask]


def remove_outliers2(data, alpha):
    """
    Remove outliers and return both the cleaned data and the indices of removed rows.

    Same logic as `remove_outliers`, but also provides indices of outliers for
    logging, debugging, or further analysis.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset.
    alpha : float
        Scaling factor for the IQR range. A typical value is 1.5.

    Returns
    -------
    cleaned_data : pandas.DataFrame
        DataFrame with outliers removed.
    removed_indices : pandas.Index
        Index of rows that were identified as outliers.
    """
    # Exclude label columns (if present) before computing IQR
    if all(col in data.columns for col in label_columns):
        data_without_labels = data.drop(columns=label_columns)
    else:
        data_without_labels = data.copy(deep=True)

    # Compute IQR bounds for each numeric column
    Q1 = data_without_labels.quantile(0.25, numeric_only=True)
    Q3 = data_without_labels.quantile(0.75, numeric_only=True)
    IQR = Q3 - Q1
    lower_bound = Q1 - alpha * IQR
    upper_bound = Q3 + alpha * IQR

    # Create mask for rows within acceptable range
    mask = ~(
        (data_without_labels.lt(lower_bound, axis=1)) |
        (data_without_labels.gt(upper_bound, axis=1))
    ).any(axis=1)

    # Indices of removed outliers
    removed_indices = data.index[~mask]

    return data[mask], removed_indices
