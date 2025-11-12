"""
Generate synthetic CPG sales dataset for testing and development.
Run this script to create the sample dataset in parquet format.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_cpg_sales_data(num_rows=5000, output_path='data/cpg_sales_data.parquet'):
    """
    Generate synthetic CPG sales data.
    
    Args:
        num_rows: Number of rows to generate
        output_path: Path to save the parquet file
    """
    np.random.seed(123)
    
    # Parameters
    date_range = pd.date_range('2022-01-01', '2024-12-31', freq='D')
    stores = list(range(1, 11))
    regions = ['North', 'South', 'East']
    skus = list(range(101, 151))
    categories = ['Beverages', 'Snacks', 'Dairy', 'Household', 'Personal Care']
    promo_types = [None, 'Discount', 'BuyOneGetOne', 'FlashSale']
    store_sizes = ['Small', 'Medium', 'Large']
    
    data = []
    
    for _ in range(num_rows):
        date = np.random.choice(date_range)
        store = np.random.choice(stores)
        store_region = np.random.choice(regions)
        sku = np.random.choice(skus)
        category = np.random.choice(categories)
        base_price = np.round(np.random.uniform(2, 10), 2)
        promo_flag = np.random.choice([0, 1], p=[0.8, 0.2])
        promo_type = np.random.choice(promo_types) if promo_flag else None
        
        # Adjust price based on promo
        if promo_flag:
            if promo_type == 'Discount':
                price = base_price * 0.8
            elif promo_type == 'BuyOneGetOne':
                price = base_price * 0.5  # Effective price
            elif promo_type == 'FlashSale':
                price = base_price * 0.7
            else:
                price = base_price
        else:
            price = base_price
        
        # Units sold affected by promo and holidays
        base_units = np.random.poisson(10)
        if promo_flag:
            base_units = int(base_units * np.random.uniform(1.5, 2.5))
        
        # Convert to pandas Timestamp for weekday check
        date_ts = pd.Timestamp(date)
        if date_ts.weekday() in [5, 6]:  # Weekend boost
            base_units = int(base_units * 1.2)
        
        units_sold = max(1, base_units)
        revenue = units_sold * price
        inventory_level = np.random.randint(100, 1000)
        store_size = np.random.choice(store_sizes)
        holiday_flag = 1 if date_ts.weekday() in [5, 6] else 0
        
        data.append([
            date, store, store_region, sku, category,
            units_sold, revenue, promo_flag, promo_type,
            price, inventory_level, store_size, holiday_flag
        ])
    
    df = pd.DataFrame(data, columns=[
        'date', 'store_id', 'store_region', 'sku_id', 'category',
        'units_sold', 'revenue', 'promo_flag', 'promo_type',
        'price', 'inventory_level', 'store_size', 'holiday_flag'
    ])
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet
    df.to_parquet(output_path, index=False)
    print(f"Dataset with {len(df)} rows created successfully at {output_path}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Stores: {df['store_id'].nunique()}, SKUs: {df['sku_id'].nunique()}")
    
    return df

if __name__ == "__main__":
    generate_cpg_sales_data()

