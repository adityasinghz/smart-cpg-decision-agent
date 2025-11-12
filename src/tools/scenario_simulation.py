"""
Logic for 'what-if' business scenario simulation.
Simulates promotional campaigns, price changes, and supply disruptions.
"""

from typing import Union, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from pyspark.sql import DataFrame as SparkDataFrame
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    SparkDataFrame = None


def simulate_promotion(
    df: Union[pd.DataFrame, SparkDataFrame],
    product_id: Optional[Union[int, str]] = None,
    store_id: Optional[Union[int, str]] = None,
    discount_pct: float = 0.15,
    duration_days: int = 7,
    start_date: Optional[str] = None,  # Reserved for future use
    price_elasticity: float = -1.5
) -> Dict:
    """
    Simulate the impact of a promotional campaign.
    
    Args:
        df: Historical sales data
        product_id: Specific product to promote (None for all products)
        store_id: Specific store (None for all stores)
        discount_pct: Discount percentage (0.15 = 15% off)
        duration_days: Duration of promotion in days
        start_date: Start date of promotion (YYYY-MM-DD), None for next available date
        price_elasticity: Price elasticity coefficient (negative = demand increases with price decrease)
    
    Returns:
        Dictionary with simulation results
    """
    if isinstance(df, SparkDataFrame):
        df = df.toPandas()
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter data
    baseline_df = df.copy()
    if product_id is not None:
        baseline_df = baseline_df[baseline_df['sku_id'] == product_id]
    if store_id is not None:
        baseline_df = baseline_df[baseline_df['store_id'] == store_id]
    
    if len(baseline_df) == 0:
        return {'error': 'No matching data found for simulation'}
    
    # Calculate baseline metrics
    baseline_revenue = baseline_df['revenue'].sum()
    baseline_units = baseline_df['units_sold'].sum()
    baseline_avg_price = baseline_df['price'].mean()
    
    # Simulate promotion impact
    # Price elasticity: % change in quantity = elasticity * % change in price
    price_change_pct = -discount_pct  # Negative because price decreases
    quantity_change_pct = price_elasticity * price_change_pct
    
    # Calculate new metrics
    new_price = baseline_avg_price * (1 - discount_pct)
    new_units = baseline_units * (1 + quantity_change_pct)
    new_revenue = new_units * new_price
    
    # Calculate incremental metrics
    incremental_units = new_units - baseline_units
    incremental_revenue = new_revenue - baseline_revenue
    revenue_lift_pct = (incremental_revenue / baseline_revenue * 100) if baseline_revenue > 0 else 0
    
    # Estimate cannibalization (some sales would have happened anyway)
    cannibalization_rate = 0.3  # 30% of promo sales would have happened anyway
    net_incremental_units = incremental_units * (1 - cannibalization_rate)
    net_incremental_revenue = incremental_revenue * (1 - cannibalization_rate)
    
    return {
        'scenario': 'promotion',
        'parameters': {
            'discount_pct': discount_pct,
            'duration_days': duration_days,
            'product_id': product_id,
            'store_id': store_id
        },
        'baseline': {
            'revenue': float(baseline_revenue),
            'units': float(baseline_units),
            'avg_price': float(baseline_avg_price)
        },
        'projected': {
            'revenue': float(new_revenue),
            'units': float(new_units),
            'avg_price': float(new_price)
        },
        'impact': {
            'incremental_revenue': float(incremental_revenue),
            'incremental_units': float(incremental_units),
            'revenue_lift_pct': float(revenue_lift_pct),
            'net_incremental_revenue': float(net_incremental_revenue),
            'net_incremental_units': float(net_incremental_units)
        },
        'recommendation': 'proceed' if revenue_lift_pct > 5 else 'review'
    }


def simulate_price_change(
    df: Union[pd.DataFrame, SparkDataFrame],
    product_id: Optional[Union[int, str]] = None,
    store_id: Optional[Union[int, str]] = None,
    price_increase_pct: float = 0.10,
    price_elasticity: float = -1.5
) -> Dict:
    """
    Simulate the impact of a price increase.
    
    Args:
        df: Historical sales data
        product_id: Specific product (None for all products)
        store_id: Specific store (None for all stores)
        price_increase_pct: Price increase percentage (0.10 = 10% increase)
        price_elasticity: Price elasticity coefficient
    
    Returns:
        Dictionary with simulation results
    """
    if isinstance(df, SparkDataFrame):
        df = df.toPandas()
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter data
    baseline_df = df.copy()
    if product_id is not None:
        baseline_df = baseline_df[baseline_df['sku_id'] == product_id]
    if store_id is not None:
        baseline_df = baseline_df[baseline_df['store_id'] == store_id]
    
    if len(baseline_df) == 0:
        return {'error': 'No matching data found for simulation'}
    
    # Calculate baseline metrics
    baseline_revenue = baseline_df['revenue'].sum()
    baseline_units = baseline_df['units_sold'].sum()
    baseline_avg_price = baseline_df['price'].mean()
    
    # Simulate price increase impact
    # Price elasticity: % change in quantity = elasticity * % change in price
    quantity_change_pct = price_elasticity * price_increase_pct
    
    # Calculate new metrics
    new_price = baseline_avg_price * (1 + price_increase_pct)
    new_units = baseline_units * (1 + quantity_change_pct)
    new_revenue = new_units * new_price
    
    # Calculate impact
    revenue_change = new_revenue - baseline_revenue
    revenue_change_pct = (revenue_change / baseline_revenue * 100) if baseline_revenue > 0 else 0
    units_change = new_units - baseline_units
    units_change_pct = (units_change / baseline_units * 100) if baseline_units > 0 else 0
    
    # Margin impact (assuming constant cost)
    margin_improvement = (new_price - baseline_avg_price) * new_units
    
    return {
        'scenario': 'price_increase',
        'parameters': {
            'price_increase_pct': price_increase_pct,
            'product_id': product_id,
            'store_id': store_id
        },
        'baseline': {
            'revenue': float(baseline_revenue),
            'units': float(baseline_units),
            'avg_price': float(baseline_avg_price)
        },
        'projected': {
            'revenue': float(new_revenue),
            'units': float(new_units),
            'avg_price': float(new_price)
        },
        'impact': {
            'revenue_change': float(revenue_change),
            'revenue_change_pct': float(revenue_change_pct),
            'units_change': float(units_change),
            'units_change_pct': float(units_change_pct),
            'margin_improvement': float(margin_improvement)
        },
        'recommendation': 'proceed' if revenue_change_pct > 0 and abs(units_change_pct) < 15 else 'review'
    }


def simulate_supply_shortage(
    df: Union[pd.DataFrame, SparkDataFrame],
    product_id: Optional[Union[int, str]] = None,
    shortage_duration_days: int = 14,
    availability_pct: float = 0.5
) -> Dict:
    """
    Simulate the impact of a supply shortage.
    
    Args:
        df: Historical sales data
        product_id: Product affected by shortage
        shortage_duration_days: Duration of shortage
        availability_pct: Percentage of normal inventory available (0.5 = 50%)
    
    Returns:
        Dictionary with simulation results
    """
    if isinstance(df, SparkDataFrame):
        df = df.toPandas()
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter data
    baseline_df = df.copy()
    if product_id is not None:
        baseline_df = baseline_df[baseline_df['sku_id'] == product_id]
    
    if len(baseline_df) == 0:
        return {'error': 'No matching data found for simulation'}
    
    # Calculate baseline metrics
    baseline_revenue = baseline_df['revenue'].sum()
    baseline_units = baseline_df['units_sold'].sum()
    
    # Simulate shortage impact
    # Lost sales = (1 - availability_pct) * baseline sales
    lost_units = baseline_units * (1 - availability_pct)
    lost_revenue = baseline_revenue * (1 - availability_pct)
    
    # Available sales
    available_units = baseline_units * availability_pct
    available_revenue = baseline_revenue * availability_pct
    
    return {
        'scenario': 'supply_shortage',
        'parameters': {
            'shortage_duration_days': shortage_duration_days,
            'availability_pct': availability_pct,
            'product_id': product_id
        },
        'baseline': {
            'revenue': float(baseline_revenue),
            'units': float(baseline_units)
        },
        'projected': {
            'revenue': float(available_revenue),
            'units': float(available_units)
        },
        'impact': {
            'lost_revenue': float(lost_revenue),
            'lost_units': float(lost_units),
            'revenue_loss_pct': float((lost_revenue / baseline_revenue * 100)) if baseline_revenue > 0 else 0
        },
        'recommendation': 'urgent_action' if availability_pct < 0.3 else 'monitor'
    }
