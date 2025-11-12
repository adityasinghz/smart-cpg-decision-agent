"""
Methods for detecting anomalies in sales data.
Uses statistical methods and time-series analysis.
"""

from typing import Union, Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

try:
    from pyspark.sql import DataFrame as SparkDataFrame
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    SparkDataFrame = None


def detect_anomalies(
    df: Union[pd.DataFrame, SparkDataFrame],
    date_col: str = 'date',
    value_col: str = 'revenue',
    method: str = 'zscore',
    threshold: float = 3.0,
    group_by: Optional[list] = None
) -> Dict:
    """
    Detect anomalies in time-series data.
    
    Args:
        df: DataFrame with time-series data
        date_col: Name of date column
        value_col: Name of value column to analyze
        method: Anomaly detection method ('zscore', 'iqr', 'isolation_forest')
        threshold: Threshold for anomaly detection (for zscore method)
        group_by: Optional list of columns to group by
    
    Returns:
        Dictionary with anomaly information
    """
    if isinstance(df, pd.DataFrame):
        return _detect_anomalies_pandas(df, date_col, value_col, method, threshold, group_by)
    else:
        pandas_df = df.toPandas()
        return _detect_anomalies_pandas(pandas_df, date_col, value_col, method, threshold, group_by)


def _detect_anomalies_pandas(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    method: str,
    threshold: float,
    group_by: Optional[list]
) -> Dict:
    """Detect anomalies using pandas."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    if group_by:
        all_anomalies = []
        for group_key, group_df in df.groupby(group_by):
            anomalies = _calculate_anomalies(group_df, date_col, value_col, method, threshold)
            anomalies['group'] = str(group_key)
            all_anomalies.append(anomalies)
        return {'grouped_anomalies': all_anomalies}
    else:
        return _calculate_anomalies(df, date_col, value_col, method, threshold)


def _calculate_anomalies(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    method: str,
    threshold: float
) -> Dict:
    """Calculate anomalies for a single time series."""
    values = df[value_col].values
    dates = df[date_col].values
    
    if method == 'zscore':
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return {'anomalies': [], 'count': 0, 'method': method}
        
        z_scores = np.abs((values - mean) / std)
        anomaly_mask = z_scores > threshold
        anomaly_indices = np.where(anomaly_mask)[0]
        
    elif method == 'iqr':
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        anomaly_mask = (values < lower_bound) | (values > upper_bound)
        anomaly_indices = np.where(anomaly_mask)[0]
        
    elif method == 'isolation_forest':
        if len(values) < 2:
            return {'anomalies': [], 'count': 0, 'method': method}
        
        # Reshape for sklearn
        X = values.reshape(-1, 1)
        clf = IsolationForest(contamination=0.1, random_state=42)
        predictions = clf.fit_predict(X)
        anomaly_mask = predictions == -1
        anomaly_indices = np.where(anomaly_mask)[0]
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    anomalies = []
    for idx in anomaly_indices:
        anomalies.append({
            'date': pd.Timestamp(dates[idx]),
            'value': float(values[idx]),
            'index': int(idx)
        })
    
    return {
        'anomalies': anomalies,
        'count': len(anomalies),
        'method': method,
        'anomaly_rate': len(anomalies) / len(values) if len(values) > 0 else 0
    }


def detect_sales_drop(
    df: Union[pd.DataFrame, SparkDataFrame],
    date_col: str = 'date',
    value_col: str = 'revenue',
    window_days: int = 7,
    drop_threshold: float = 0.2
) -> Dict:
    """
    Detect significant sales drops compared to recent average.
    
    Args:
        df: DataFrame with sales data
        date_col: Name of date column
        value_col: Name of value column
        window_days: Number of days to look back for comparison
        drop_threshold: Percentage drop to consider significant (0.2 = 20%)
    
    Returns:
        Dictionary with detected sales drops
    """
    if isinstance(df, SparkDataFrame):
        df = df.toPandas()
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    # Calculate rolling average
    df['rolling_avg'] = df[value_col].rolling(window=window_days, min_periods=1).mean()
    df['pct_change'] = (df[value_col] - df['rolling_avg']) / df['rolling_avg']
    
    # Detect significant drops
    drops = df[df['pct_change'] < -drop_threshold].copy()
    
    return {
        'drops': drops[[date_col, value_col, 'rolling_avg', 'pct_change']].to_dict('records'),
        'count': len(drops),
        'severity': 'high' if len(drops) > 10 else 'moderate' if len(drops) > 5 else 'low'
    }
