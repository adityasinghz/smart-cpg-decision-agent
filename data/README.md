# Data Directory

This directory should contain:

- `cpg_sales_data.parquet` - Synthetic big dataset in parquet format

## Expected Data Schema

The parquet file should contain CPG sales data with columns such as:
- `date` - Date/time column
- `product_id` - Product identifier
- `sales` - Sales value/volume
- `price` - Product price
- `region` - Geographic region
- `channel` - Sales channel
- Other relevant CPG metrics

## Generating Sample Data

You can create synthetic data using pandas:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample data
dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
n_products = 100
n_regions = 10

data = []
for date in dates:
    for product_id in range(1, n_products + 1):
        for region in range(1, n_regions + 1):
            data.append({
                'date': date,
                'product_id': f'PROD{product_id:03d}',
                'region': f'REGION{region:02d}',
                'sales': np.random.normal(1000, 200),
                'price': np.random.normal(10, 2),
                'units': np.random.poisson(100)
            })

df = pd.DataFrame(data)
df.to_parquet('cpg_sales_data.parquet', index=False)
```

