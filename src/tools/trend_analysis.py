"""
Functions for trend and seasonality extraction from time-series sales data.
Supports both PySpark and pandas DataFrames.
"""

from typing import Union, Dict, Optional
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

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
    method: str = 'linear',
    period: str = 'daily'
) -> Dict:
    """
    Extract trends from time-series data matching reference implementation.
    
    Args:
        df: DataFrame with time-series data
        date_col: Name of date column
        value_col: Name of value column to analyze
        group_by: Optional list of columns to group by (e.g., ['store_id', 'sku_id'])
        method: Trend extraction method ('linear', 'moving_average')
        period: Aggregation period ('daily', 'weekly', 'monthly') - matching reference
    
    Returns:
        Dictionary with trend information matching reference format
    """
    if isinstance(df, pd.DataFrame):
        return _extract_trends_pandas(df, date_col, value_col, group_by, method, period)
    else:
        return _extract_trends_spark(df, date_col, value_col, group_by, method, period)


def _extract_trends_pandas(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_by: Optional[list],
    method: str,
    period: str = 'daily'
) -> Dict:
    """Extract trends using pandas, matching reference implementation."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    if group_by:
        results = {}
        for group_key, group_df in df.groupby(group_by):
            trend_data = _calculate_trend(group_df, date_col, value_col, method, period)
            results[str(group_key)] = trend_data
        return results
    else:
        return _calculate_trend(df, date_col, value_col, method, period)


def _aggregate_by_period(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    period: str
) -> pd.DataFrame:
    """Aggregate data by time period matching reference implementation."""
    df = df.copy()
    
    # Create period column
    if period == 'daily':
        df['period_start'] = df[date_col]
    elif period == 'weekly':
        df['period_start'] = df[date_col] - pd.to_timedelta(df[date_col].dt.dayofweek, unit='d')
    elif period == 'monthly':
        df['period_start'] = df[date_col].dt.to_period('M').dt.to_timestamp()
    elif period == 'quarterly':
        df['period_start'] = df[date_col].dt.to_period('Q').dt.to_timestamp()
    else:
        raise ValueError(f"Unsupported period: {period}")
    
    # Aggregate by period (sum values)
    aggregated = df.groupby('period_start')[value_col].sum().reset_index()
    aggregated = aggregated.sort_values('period_start').reset_index(drop=True)
    
    return aggregated


def _calculate_trend(df: pd.DataFrame, date_col: str, value_col: str, method: str, period: str = 'daily') -> Dict:
    """Calculate trend for a single time series matching reference implementation."""
    df = df.sort_values(date_col)
    
    # Aggregate by period first (matching reference)
    aggregated = _aggregate_by_period(df, date_col, value_col, period)
    
    if len(aggregated) < 3:
        return {
            'trend': 'insufficient_data',
            'slope': 0,
            'r_squared': 0,
            'trend_direction': 'flat',
            'trend_strength': 'weak'
        }
    
    values = aggregated[value_col].values
    
    if method == 'linear':
        # Prepare data for regression (matching reference: X = np.arange(len(aggregated)))
        X = np.arange(len(aggregated)).reshape(-1, 1)
        y = values
        
        # Fit linear regression using sklearn (matching reference)
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate R²
        r_squared = model.score(X, y)
        
        # Statistical significance test (matching reference)
        try:
            _, p_value = stats.pearsonr(X.flatten(), y)
        except:
            p_value = 1.0
        
        # Determine trend direction (matching reference logic)
        slope = model.coef_[0]
        if abs(slope) < 0.01:
            trend_direction = 'flat'
        elif slope > 0:
            trend_direction = 'increasing'
        else:
            trend_direction = 'decreasing'
        
        # Calculate percentage change (matching reference)
        start_value = y[0]
        end_value = y[-1]
        pct_change = ((end_value - start_value) / start_value) * 100 if start_value != 0 else 0
        
        # Trend strength (matching reference)
        if abs(r_squared) > 0.7:
            trend_strength = 'strong'
        elif abs(r_squared) > 0.4:
            trend_strength = 'moderate'
        else:
            trend_strength = 'weak'
        
        return {
            'trend': trend_direction,
            'slope': float(slope),
            'intercept': float(model.intercept_),
            'r_squared': float(r_squared),
            'p_value': float(p_value),
            'is_significant': p_value < 0.05,
            'percentage_change': float(pct_change),
            'start_value': float(start_value),
            'end_value': float(end_value),
            'data_points': len(aggregated),
            'period': period,
            'trend_direction': trend_direction,  # For backward compatibility
            'trend_strength': trend_strength
        }
    elif method == 'moving_average':
        # Moving average trend
        window_size = min(30, len(aggregated) // 4)
        if window_size < 1:
            window_size = 1
        ma = pd.Series(values).rolling(window=window_size, center=True).mean()
        
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
    method: str,
    period: str = 'daily'
) -> Dict:
    """Extract trends using PySpark."""
    # For PySpark, convert to pandas for trend calculation
    # In production, you'd implement native Spark operations
    pandas_df = df.toPandas()
    return _extract_trends_pandas(pandas_df, date_col, value_col, group_by, method, period)


def calculate_growth_rate(
    df: Union[pd.DataFrame, SparkDataFrame],
    date_col: str = 'date',
    value_col: str = 'revenue',
    period: str = 'monthly',
    method: str = 'compound'
) -> Dict:
    """
    Calculate growth rate over time matching reference implementation.
    
    Args:
        df: DataFrame with time-series data
        date_col: Name of date column
        value_col: Name of value column to analyze
        period: Aggregation period ('weekly', 'monthly', 'quarterly')
        method: 'compound' for CAGR or 'simple' for average growth
    
    Returns:
        Dictionary with growth rate statistics matching reference format
    """
    if isinstance(df, pd.DataFrame):
        return _calculate_growth_rate_pandas(df, date_col, value_col, period, method)
    else:
        pandas_df = df.toPandas()
        return _calculate_growth_rate_pandas(pandas_df, date_col, value_col, period, method)


def _calculate_growth_rate_pandas(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    period: str,
    method: str
) -> Dict:
    """Calculate growth rate using pandas, matching reference implementation."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    # Aggregate by period
    aggregated = _aggregate_by_period(df, date_col, value_col, period)
    
    if len(aggregated) < 2:
        return {
            'growth_rate': 0,
            'method': method,
            'periods': len(aggregated)
        }
    
    values = aggregated[value_col].values
    
    if method == 'compound':
        # Compound Annual Growth Rate (CAGR) - matching reference
        start_value = values[0]
        end_value = values[-1]
        n_periods = len(values) - 1
        
        if start_value <= 0:
            growth_rate = 0
        else:
            # Calculate period-over-period growth rate
            growth_rate = (((end_value / start_value) ** (1 / n_periods)) - 1) * 100
    else:
        # Simple average period-over-period growth
        period_growth = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                growth = ((values[i] - values[i-1]) / values[i-1]) * 100
                period_growth.append(growth)
        
        growth_rate = np.mean(period_growth) if period_growth else 0
    
    return {
        'growth_rate': float(growth_rate),
        'method': method,
        'period': period,
        'periods': len(values),
        'start_value': float(values[0]),
        'end_value': float(values[-1]),
        'metric': value_col
    }


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
