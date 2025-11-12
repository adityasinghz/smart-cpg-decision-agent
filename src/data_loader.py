"""
Data ingestion utilities using PySpark and pandas.
Supports both Databricks (PySpark) and local (pandas) environments.
"""

import os
from typing import Union, Optional
import pandas as pd

try:
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from pyspark.sql.functions import col, to_date
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    SparkSession = None
    SparkDataFrame = None


def get_spark_session() -> Optional[SparkSession]:
    """Get or create a Spark session for Databricks."""
    if not PYSPARK_AVAILABLE:
        return None
    
    try:
        # Try to get existing Spark session (Databricks)
        spark = SparkSession.getActiveSession()
        if spark is None:
            # Create new session if none exists
            spark = SparkSession.builder \
                .appName("CPGDecisionAgent") \
                .getOrCreate()
        return spark
    except Exception:
        return None


def load_cpg_data(
    file_path: str,
    use_spark: Optional[bool] = None,
    date_col: str = 'date'
) -> Union[pd.DataFrame, SparkDataFrame]:
    """
    Load CPG sales data from parquet file.
    
    Args:
        file_path: Path to parquet file
        use_spark: If True, use PySpark. If False, use pandas. If None, auto-detect.
        date_col: Name of the date column to parse
    
    Returns:
        DataFrame (pandas or PySpark)
    """
    if use_spark is None:
        # Auto-detect: use Spark if available and in Databricks environment
        use_spark = PYSPARK_AVAILABLE and os.getenv('DATABRICKS_RUNTIME_VERSION') is not None
    
    if use_spark and PYSPARK_AVAILABLE:
        spark = get_spark_session()
        if spark is None:
            raise RuntimeError("PySpark requested but Spark session unavailable")
        
        df = spark.read.parquet(file_path)
        
        # Convert date column if needed
        if date_col in df.columns:
            df = df.withColumn(date_col, to_date(col(date_col)))
        
        return df
    else:
        # Use pandas
        df = pd.read_parquet(file_path)
        
        # Convert date column if needed
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
        
        return df


def convert_spark_to_pandas(df: SparkDataFrame) -> pd.DataFrame:
    """Convert PySpark DataFrame to pandas DataFrame."""
    if not PYSPARK_AVAILABLE:
        raise ImportError("PySpark not available")
    return df.toPandas()


def get_data_summary(df: Union[pd.DataFrame, SparkDataFrame]) -> dict:
    """
    Get summary statistics of the dataset.
    
    Args:
        df: DataFrame (pandas or PySpark)
    
    Returns:
        Dictionary with summary statistics
    """
    if isinstance(df, pd.DataFrame):
        return {
            'rows': len(df),
            'columns': list(df.columns),
            'date_range': (df['date'].min(), df['date'].max()) if 'date' in df.columns else None,
            'stores': df['store_id'].nunique() if 'store_id' in df.columns else None,
            'skus': df['sku_id'].nunique() if 'sku_id' in df.columns else None,
            'total_revenue': df['revenue'].sum() if 'revenue' in df.columns else None,
        }
    else:
        # PySpark DataFrame
        from pyspark.sql.functions import min as spark_min, max as spark_max, sum as spark_sum, countDistinct
        summary = df.select(
            spark_min('date').alias('min_date'),
            spark_max('date').alias('max_date'),
            countDistinct('store_id').alias('num_stores'),
            countDistinct('sku_id').alias('num_skus'),
            spark_sum('revenue').alias('total_revenue')
        ).collect()[0]
        
        return {
            'rows': df.count(),
            'columns': df.columns,
            'date_range': (summary['min_date'], summary['max_date']) if summary['min_date'] else None,
            'stores': summary['num_stores'],
            'skus': summary['num_skus'],
            'total_revenue': summary['total_revenue'],
        }
