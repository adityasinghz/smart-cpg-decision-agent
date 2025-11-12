"""
Functions for trend and seasonality extraction from time-series sales data.
Supports both PySpark and pandas DataFrames.
"""

from typing import Union, Dict, Optional
import pandas as pd
import numpy as np

try:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql.functions import col, date_format, avg, sum as spark_sum, window
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    SparkDataFrame = None


def extract_trends(
    df: Union[pd.DataFrame, SparkDataFrame],
    date_col: str = 'date',
    value_col: str = 'revenue',
    group_by: Optional[list] = None,
    method: str = 'linear'
) -> Dict:
    """
    Extract trends from time-series data.
    
    Args:
        df: DataFrame with time-series data
        date_col: Name of date column
        value_col: Name of value column to analyze
        group_by: Optional list of columns to group by (e.g., ['store_id', 'sku_id'])
        method: Trend extraction method ('linear', 'moving_average')
    
    Returns:
        Dictionary with trend information
    """
    if isinstance(df, pd.DataFrame):
        return _extract_trends_pandas(df, date_col, value_col, group_by, method)
    else:
        return _extract_trends_spark(df, date_col, value_col, group_by, method)


def _extract_trends_pandas(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_by: Optional[list],
    method: str
) -> Dict:
    """Extract trends using pandas."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    if group_by:
        results = {}
        for group_key, group_df in df.groupby(group_by):
            trend_data = _calculate_trend(group_df, date_col, value_col, method)
            results[str(group_key)] = trend_data
        return results
    else:
        return _calculate_trend(df, date_col, value_col, method)


def _calculate_trend(df: pd.DataFrame, date_col: str, value_col: str, method: str) -> Dict:
    """Calculate trend for a single time series."""
    df = df.sort_values(date_col)
    dates = df[date_col]
    values = df[value_col]
    
    # Convert dates to numeric for regression
    date_numeric = (dates - dates.min()).dt.days
    
    if method == 'linear':
        # Linear regression
        coeffs = np.polyfit(date_numeric, values, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        trend_line = np.polyval(coeffs, date_numeric)
        
        # Calculate R-squared
        ss_res = np.sum((values - trend_line) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'trend_direction': 'increasing' if slope > 0 else 'decreasing',
            'trend_strength': 'strong' if abs(r_squared) > 0.7 else 'moderate' if abs(r_squared) > 0.4 else 'weak'
        }
    elif method == 'moving_average':
        # Moving average trend
        window_size = min(30, len(df) // 4)
        if window_size < 1:
            window_size = 1
        ma = values.rolling(window=window_size, center=True).mean()
        
        # Calculate overall trend direction
        first_half = ma.iloc[:len(ma)//2].mean()
        second_half = ma.iloc[len(ma)//2:].mean()
        slope = (second_half - first_half) / len(ma)
        
        return {
            'slope': float(slope),
            'moving_average': ma.tolist(),
            'trend_direction': 'increasing' if slope > 0 else 'decreasing'
        }
    else:
        raise ValueError(f"Unknown method: {method}")


def _extract_trends_spark(
    df: SparkDataFrame,
    date_col: str,
    value_col: str,
    group_by: Optional[list],
    method: str
) -> Dict:
    """Extract trends using PySpark."""
    # For PySpark, convert to pandas for trend calculation
    # In production, you'd implement native Spark operations
    pandas_df = df.toPandas()
    return _extract_trends_pandas(pandas_df, date_col, value_col, group_by, method)


def detect_seasonality(
    df: Union[pd.DataFrame, SparkDataFrame],
    date_col: str = 'date',
    value_col: str = 'revenue',
    period: str = 'weekly',
    group_by: Optional[list] = None
) -> Dict:
    """
    Detect seasonality patterns in time-series data.
    
    Args:
        df: DataFrame with time-series data
        date_col: Name of date column
        value_col: Name of value column to analyze
        period: Seasonality period ('weekly', 'monthly', 'quarterly')
        group_by: Optional list of columns to group by
    
    Returns:
        Dictionary with seasonality information
    """
    if isinstance(df, pd.DataFrame):
        return _detect_seasonality_pandas(df, date_col, value_col, period, group_by)
    else:
        pandas_df = df.toPandas()
        return _detect_seasonality_pandas(pandas_df, date_col, value_col, period, group_by)


def _detect_seasonality_pandas(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    period: str,
    group_by: Optional[list]
) -> Dict:
    """Detect seasonality using pandas."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    if period == 'weekly':
        df['period'] = df[date_col].dt.dayofweek
        period_name = 'day_of_week'
    elif period == 'monthly':
        df['period'] = df[date_col].dt.day
        period_name = 'day_of_month'
    elif period == 'quarterly':
        df['period'] = df[date_col].dt.quarter
        period_name = 'quarter'
    else:
        raise ValueError(f"Unknown period: {period}")
    
    if group_by:
        results = {}
        for group_key, group_df in df.groupby(group_by):
            seasonality = group_df.groupby('period')[value_col].mean().to_dict()
            results[str(group_key)] = {
                period_name: seasonality,
                'seasonality_strength': _calculate_seasonality_strength(list(seasonality.values()))
            }
        return results
    else:
        seasonality = df.groupby('period')[value_col].mean().to_dict()
        return {
            period_name: seasonality,
            'seasonality_strength': _calculate_seasonality_strength(list(seasonality.values()))
        }


def _calculate_seasonality_strength(values: list) -> str:
    """Calculate strength of seasonality pattern."""
    if not values:
        return 'none'
    
    cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
    
    if cv > 0.3:
        return 'strong'
    elif cv > 0.15:
        return 'moderate'
    else:
        return 'weak'


def compare_stores(
    df: Union[pd.DataFrame, SparkDataFrame],
    date_col: str = 'date',
    value_col: str = 'revenue',
    store_col: str = 'store_id'
) -> Dict:
    """
    Compare performance across stores.
    
    Args:
        df: DataFrame with sales data
        date_col: Name of date column
        value_col: Name of value column to compare
        store_col: Name of store identifier column
    
    Returns:
        Dictionary with store comparison metrics
    """
    if isinstance(df, pd.DataFrame):
        store_stats = df.groupby(store_col).agg({
            value_col: ['sum', 'mean', 'std', 'count']
        }).reset_index()
        store_stats.columns = [store_col, 'total', 'average', 'std_dev', 'count']
        
        return {
            'store_performance': store_stats.to_dict('records'),
            'top_store': store_stats.loc[store_stats['total'].idxmax(), store_col],
            'bottom_store': store_stats.loc[store_stats['total'].idxmin(), store_col]
        }
    else:
        pandas_df = df.toPandas()
        return compare_stores(pandas_df, date_col, value_col, store_col)
