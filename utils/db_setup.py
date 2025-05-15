"""
Database setup and sample data generation for the Supply Chain Analytics tool.
This module contains utilities for setting up the MySQL database schema and
generating sample data for testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_shipment_data(num_records=1000):
    """
    Generate sample shipment data.
    
    Parameters:
    -----------
    num_records : int
        Number of sample records to generate
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing sample shipment data
    """
    # Define possible values
    areas = ['Chicago', 'Dallas', 'Houston', 'Los Angeles', 'New York', 'Philadelphia', 'Phoenix', 'San Antonio']
    product_categories = ['Electronics', 'Clothing', 'Food', 'Furniture', 'Toys']
    transportation_modes = ['Truck', 'Rail', 'Air', 'Ship']
    
    # Generate random data
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    shipment_dates = [start_date + timedelta(days=random.randint(0, 365)) for _ in range(num_records)]
    areas_sampled = [random.choice(areas) for _ in range(num_records)]
    product_categories_sampled = [random.choice(product_categories) for _ in range(num_records)]
    quantities = [random.randint(100, 1000) for _ in range(num_records)]
    shipment_costs = [round(random.uniform(500, 5000), 2) for _ in range(num_records)]
    transportation_modes_sampled = [random.choice(transportation_modes) for _ in range(num_records)]
    delivery_times = [round(random.uniform(1, 15), 1) for _ in range(num_records)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'shipment_date': shipment_dates,
        'area': areas_sampled,
        'product_category': product_categories_sampled,
        'quantity_sent': quantities,
        'shipment_cost': shipment_costs,
        'transportation_mode': transportation_modes_sampled,
        'delivery_time_days': delivery_times
    })
    
    return df

def generate_sample_sales_data(num_records=1000):
    """
    Generate sample sales data.
    
    Parameters:
    -----------
    num_records : int
        Number of sample records to generate
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing sample sales data
    """
    # Define possible values
    areas = ['Chicago', 'Dallas', 'Houston', 'Los Angeles', 'New York', 'Philadelphia', 'Phoenix', 'San Antonio']
    product_categories = ['Electronics', 'Clothing', 'Food', 'Furniture', 'Toys']
    customer_segments = ['Retail', 'Wholesale', 'Online', 'Corporate']
    
    # Generate random data
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    sale_dates = [start_date + timedelta(days=random.randint(0, 365)) for _ in range(num_records)]
    areas_sampled = [random.choice(areas) for _ in range(num_records)]
    product_categories_sampled = [random.choice(product_categories) for _ in range(num_records)]
    units_sold = [random.randint(50, 800) for _ in range(num_records)]
    revenues = [round(random.uniform(1000, 10000), 2) for _ in range(num_records)]
    customer_segments_sampled = [random.choice(customer_segments) for _ in range(num_records)]
    profit_margins = [round(random.uniform(0.1, 0.4), 2) for _ in range(num_records)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'sale_date': sale_dates,
        'area': areas_sampled,
        'product_category': product_categories_sampled,
        'units_sold': units_sold,
        'revenue': revenues,
        'customer_segment': customer_segments_sampled,
        'profit_margin': profit_margins
    })
    
    return df

def generate_sample_inventory_data(num_records=200):
    """
    Generate sample inventory data.
    
    Parameters:
    -----------
    num_records : int
        Number of sample records to generate
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing sample inventory data
    """
    # Define possible values
    areas = ['Chicago', 'Dallas', 'Houston', 'Los Angeles', 'New York', 'Philadelphia', 'Phoenix', 'San Antonio']
    product_categories = ['Electronics', 'Clothing', 'Food', 'Furniture', 'Toys']
    
    # Generate random data
    start_date = datetime(2023, 1, 1)
    dates = []
    areas_sampled = []
    product_categories_sampled = []
    
    # Create combinations of dates, areas, and product categories
    for area in areas:
        for product_category in product_categories:
            for month in range(12):
                dates.append(start_date + timedelta(days=30*month))
                areas_sampled.append(area)
                product_categories_sampled.append(product_category)
    
    # Limit to num_records if needed
    if len(dates) > num_records:
        indices = random.sample(range(len(dates)), num_records)
        dates = [dates[i] for i in indices]
        areas_sampled = [areas_sampled[i] for i in indices]
        product_categories_sampled = [product_categories_sampled[i] for i in indices]
    
    remaining_stocks = [random.randint(100, 2000) for _ in range(len(dates))]
    holding_costs = [round(random.uniform(200, 2000), 2) for _ in range(len(dates))]
    stockout_counts = [random.randint(0, 10) for _ in range(len(dates))]
    days_of_supply = [random.randint(15, 60) for _ in range(len(dates))]
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'area': areas_sampled,
        'product_category': product_categories_sampled,
        'remaining_stock': remaining_stocks,
        'holding_cost': holding_costs,
        'stockout_count': stockout_counts,
        'days_of_supply': days_of_supply
    })
    
    return df

def generate_sample_cost_data(num_records=200):
    """
    Generate sample cost data.
    
    Parameters:
    -----------
    num_records : int
        Number of sample records to generate
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing sample cost data
    """
    # Define possible values
    areas = ['Chicago', 'Dallas', 'Houston', 'Los Angeles', 'New York', 'Philadelphia', 'Phoenix', 'San Antonio']
    
    # Generate random data
    start_date = datetime(2023, 1, 1)
    dates = []
    areas_sampled = []
    
    # Create combinations of dates and areas
    for area in areas:
        for month in range(12):
            dates.append(start_date + timedelta(days=30*month))
            areas_sampled.append(area)
    
    # Limit to num_records if needed
    if len(dates) > num_records:
        indices = random.sample(range(len(dates)), num_records)
        dates = [dates[i] for i in indices]
        areas_sampled = [areas_sampled[i] for i in indices]
    
    transportation_costs = [round(random.uniform(5000, 15000), 2) for _ in range(len(dates))]
    warehouse_costs = [round(random.uniform(3000, 10000), 2) for _ in range(len(dates))]
    handling_costs = [round(random.uniform(1000, 5000), 2) for _ in range(len(dates))]
    other_costs = [round(random.uniform(500, 3000), 2) for _ in range(len(dates))]
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'area': areas_sampled,
        'transportation_cost': transportation_costs,
        'warehouse_cost': warehouse_costs,
        'handling_cost': handling_costs,
        'other_costs': other_costs
    })
    
    return df

def generate_all_sample_data():
    """
    Generate all sample datasets.
    
    Returns:
    --------
    tuple
        Tuple containing (shipment_df, sales_df, inventory_df, cost_df)
    """
    shipment_df = generate_sample_shipment_data()
    sales_df = generate_sample_sales_data()
    inventory_df = generate_sample_inventory_data()
    cost_df = generate_sample_cost_data()
    
    return shipment_df, sales_df, inventory_df, cost_df