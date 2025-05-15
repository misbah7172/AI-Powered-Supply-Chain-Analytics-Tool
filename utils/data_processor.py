import pandas as pd
import numpy as np
from datetime import datetime

class DataProcessor:
    """
    Class for processing and transforming data for the supply chain analytics tool.
    Handles data cleaning, preprocessing, and merging from multiple sources.
    """
    
    def __init__(self, shipment_data, sales_data, inventory_data, cost_data):
        """
        Initialize the data processor with the input dataframes.
        
        Parameters:
        -----------
        shipment_data : pandas.DataFrame
            Data containing shipment information (area, date, quantity sent, shipment cost)
        sales_data : pandas.DataFrame
            Data containing sales information (area, units sold, revenue)
        inventory_data : pandas.DataFrame
            Data containing inventory information (area, remaining stock)
        cost_data : pandas.DataFrame
            Data containing cost information (area, transportation cost, warehouse cost)
        """
        self.shipment_data = shipment_data
        self.sales_data = sales_data
        self.inventory_data = inventory_data
        self.cost_data = cost_data
        
        # Preprocess the data
        self._preprocess_data()
        
    def _preprocess_data(self):
        """
        Preprocess the data by:
        1. Handling missing values
        2. Converting date/time formats
        3. Standardizing area names
        4. Normalizing numeric values
        """
        # Make copies to avoid modifying original data
        self.shipment_data = self.shipment_data.copy()
        self.sales_data = self.sales_data.copy()
        self.inventory_data = self.inventory_data.copy()
        self.cost_data = self.cost_data.copy()
        
        # Handle missing values
        self._handle_missing_values()
        
        # Convert date formats
        self._convert_date_formats()
        
        # Standardize area names
        self._standardize_area_names()
        
        # Normalize numeric values
        self._normalize_numeric_values()
        
    def _handle_missing_values(self):
        """Handle missing values in all dataframes."""
        # Fill missing values with appropriate strategies
        # For numeric columns, fill with mean or median
        # For categorical columns, fill with mode or a default value
        
        # Shipment data
        if 'quantity_sent' in self.shipment_data.columns:
            self.shipment_data['quantity_sent'] = self.shipment_data['quantity_sent'].fillna(
                self.shipment_data['quantity_sent'].median())
        if 'shipment_cost' in self.shipment_data.columns:
            self.shipment_data['shipment_cost'] = self.shipment_data['shipment_cost'].fillna(
                self.shipment_data['shipment_cost'].median())
        
        # Sales data
        if 'units_sold' in self.sales_data.columns:
            self.sales_data['units_sold'] = self.sales_data['units_sold'].fillna(
                self.sales_data['units_sold'].median())
        if 'revenue' in self.sales_data.columns:
            self.sales_data['revenue'] = self.sales_data['revenue'].fillna(
                self.sales_data['revenue'].median())
        
        # Inventory data
        if 'remaining_stock' in self.inventory_data.columns:
            self.inventory_data['remaining_stock'] = self.inventory_data['remaining_stock'].fillna(
                self.inventory_data['remaining_stock'].median())
        
        # Cost data
        numeric_cost_cols = self.cost_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cost_cols:
            self.cost_data[col] = self.cost_data[col].fillna(self.cost_data[col].median())
        
    def _convert_date_formats(self):
        """Convert date/time formats to datetime objects."""
        # Convert date columns to datetime format
        date_columns = ['date', 'shipment_date', 'sales_date', 'inventory_date']
        
        for df in [self.shipment_data, self.sales_data, self.inventory_data, self.cost_data]:
            for col in date_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        # If conversion fails, try common formats
                        try:
                            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
                        except:
                            pass
    
    def _standardize_area_names(self):
        """Standardize area names across all dataframes."""
        # Get all area names from all dataframes
        area_columns = ['area', 'area_name', 'region', 'location']
        all_areas = set()
        
        for df in [self.shipment_data, self.sales_data, self.inventory_data, self.cost_data]:
            for col in area_columns:
                if col in df.columns:
                    all_areas.update(df[col].dropna().unique())
        
        # Create a mapping of area names to standardized names
        area_mapping = {}
        for area in all_areas:
            # Convert to lowercase and strip whitespace for standardization
            standardized = str(area).lower().strip()
            area_mapping[area] = standardized
        
        # Apply standardization to all dataframes
        for df in [self.shipment_data, self.sales_data, self.inventory_data, self.cost_data]:
            for col in area_columns:
                if col in df.columns:
                    df[col] = df[col].map(lambda x: area_mapping.get(x, x))
    
    def _normalize_numeric_values(self):
        """Normalize numeric values to ensure consistency."""
        # For each dataframe, normalize numeric columns
        for df in [self.shipment_data, self.sales_data, self.inventory_data, self.cost_data]:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # Check if values are positive (for ratio scaling)
                if df[col].min() >= 0:
                    # Skip columns that shouldn't be normalized (like IDs or counts)
                    if not any(skip in col.lower() for skip in ['id', 'count', 'number']):
                        # Min-max normalization for values that should be in range [0,1]
                        if 'ratio' in col.lower() or 'rate' in col.lower() or 'percentage' in col.lower():
                            min_val = df[col].min()
                            max_val = df[col].max()
                            if max_val > min_val:  # Avoid division by zero
                                df[col] = (df[col] - min_val) / (max_val - min_val)
    
    def get_merged_data(self):
        """
        Merge all dataframes based on common fields (area and date if available).
        
        Returns:
        --------
        pandas.DataFrame
            A merged dataframe with data from all sources.
        """
        # Identify common columns for merging
        merge_on = []
        
        # Check for area column variants
        for area_col in ['area', 'area_name', 'region', 'location']:
            if (area_col in self.shipment_data.columns and 
                area_col in self.sales_data.columns and 
                area_col in self.inventory_data.columns and 
                area_col in self.cost_data.columns):
                merge_on.append(area_col)
                break
        
        # Check for date column variants
        for date_col in ['date', 'shipment_date', 'sales_date']:
            if (date_col in self.shipment_data.columns and 
                date_col in self.sales_data.columns):
                merge_on.append(date_col)
                break
        
        # If common columns found, merge on them
        if merge_on:
            # Merge shipment and sales data
            merged_data = pd.merge(
                self.shipment_data, 
                self.sales_data,
                on=merge_on,
                how='outer'
            )
            
            # Merge with inventory data (may only have area, not date)
            if len(merge_on) > 1:  # If we have both area and date
                # Try to merge on both
                try:
                    merged_data = pd.merge(
                        merged_data,
                        self.inventory_data,
                        on=merge_on,
                        how='outer'
                    )
                except:
                    # If fails, merge only on area
                    merged_data = pd.merge(
                        merged_data,
                        self.inventory_data,
                        on=[merge_on[0]],  # Just the area column
                        how='outer'
                    )
            else:  # If we only have area
                merged_data = pd.merge(
                    merged_data,
                    self.inventory_data,
                    on=merge_on,
                    how='outer'
                )
            
            # Merge with cost data
            merged_data = pd.merge(
                merged_data,
                self.cost_data,
                on=[merge_on[0]],  # Just the area column
                how='outer'
            )
            
            return merged_data
        else:
            # If no common columns, return None and handle this case elsewhere
            return None
    
    def get_areas(self):
        """
        Get a list of unique areas from all data sources.
        
        Returns:
        --------
        list
            A list of unique area names.
        """
        areas = set()
        
        # Check for area column variants
        for df in [self.shipment_data, self.sales_data, self.inventory_data, self.cost_data]:
            for col in ['area', 'area_name', 'region', 'location']:
                if col in df.columns:
                    areas.update(df[col].dropna().unique())
        
        return sorted(list(areas))
    
    def get_date_range(self):
        """
        Get the min and max dates from the data.
        
        Returns:
        --------
        tuple
            A tuple containing (min_date, max_date).
        """
        dates = []
        
        # Check for date column variants
        for df in [self.shipment_data, self.sales_data]:
            for col in ['date', 'shipment_date', 'sales_date']:
                if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
                    dates.extend(df[col].dropna().tolist())
        
        if dates:
            return min(dates), max(dates)
        else:
            return None, None
    
    def calculate_profitability(self):
        """
        Calculate profitability metrics for each area.
        
        Returns:
        --------
        pandas.DataFrame
            A dataframe with profitability metrics by area.
        """
        # Check for required columns
        required_columns = {
            'revenue': ['revenue', 'sales_amount', 'total_revenue'],
            'shipment_cost': ['shipment_cost', 'logistics_cost', 'transportation_cost'],
            'warehouse_cost': ['warehouse_cost', 'storage_cost'],
            'area': ['area', 'area_name', 'region', 'location']
        }
        
        # Get the actual column names from the dataframes
        column_mapping = {}
        for key, variants in required_columns.items():
            for df in [self.sales_data, self.shipment_data, self.cost_data]:
                for variant in variants:
                    if variant in df.columns:
                        column_mapping[key] = variant
                        break
                if key in column_mapping:
                    break
        
        # If we don't have all required columns, return None
        if len(column_mapping) < 3:  # Need at least revenue, cost, and area
            return None
        
        # Create a merged dataset for profitability calculation
        merged_data = self.get_merged_data()
        if merged_data is None:
            return None
        
        # Calculate profitability
        results = []
        for area in self.get_areas():
            area_data = merged_data[merged_data[column_mapping['area']] == area]
            
            # Calculate metrics
            total_revenue = area_data[column_mapping.get('revenue', 'revenue')].sum() if column_mapping.get('revenue') in area_data.columns else 0
            total_shipment_cost = area_data[column_mapping.get('shipment_cost', 'shipment_cost')].sum() if column_mapping.get('shipment_cost') in area_data.columns else 0
            total_warehouse_cost = area_data[column_mapping.get('warehouse_cost', 'warehouse_cost')].sum() if column_mapping.get('warehouse_cost') in area_data.columns else 0
            
            total_cost = total_shipment_cost + total_warehouse_cost
            profit = total_revenue - total_cost
            profit_margin = (profit / total_revenue * 100) if total_revenue > 0 else 0
            
            results.append({
                'area': area,
                'total_revenue': total_revenue,
                'total_cost': total_cost,
                'profit': profit,
                'profit_margin': profit_margin
            })
        
        return pd.DataFrame(results)
    
    def get_time_series_data(self, area=None):
        """
        Get time series data for visualization and forecasting.
        
        Parameters:
        -----------
        area : str, optional
            If provided, filter data for this specific area.
            
        Returns:
        --------
        pandas.DataFrame
            A dataframe with time series data.
        """
        # Get date column
        date_col = None
        for col in ['date', 'shipment_date', 'sales_date']:
            if col in self.sales_data.columns:
                date_col = col
                break
        
        if date_col is None:
            return None
            
        # Get area column
        area_col = None
        for col in ['area', 'area_name', 'region', 'location']:
            if col in self.sales_data.columns:
                area_col = col
                break
        
        if area_col is None:
            return None
        
        # Create a copy of sales data
        time_series = self.sales_data.copy()
        
        # Ensure date column is datetime
        time_series[date_col] = pd.to_datetime(time_series[date_col])
        
        # Filter by area if specified
        if area is not None:
            time_series = time_series[time_series[area_col] == area]
        
        # Sort by date
        time_series = time_series.sort_values(by=date_col)
        
        return time_series
    
    def get_shipment_efficiency(self):
        """
        Calculate shipment efficiency metrics.
        
        Returns:
        --------
        pandas.DataFrame
            A dataframe with shipment efficiency metrics.
        """
        # Check for required columns
        required_columns = {
            'quantity_sent': ['quantity_sent', 'shipment_quantity', 'units_shipped'],
            'units_sold': ['units_sold', 'sales_quantity', 'quantity_sold'],
            'area': ['area', 'area_name', 'region', 'location']
        }
        
        # Get the actual column names from the dataframes
        column_mapping = {}
        for key, variants in required_columns.items():
            for df in [self.shipment_data, self.sales_data]:
                for variant in variants:
                    if variant in df.columns:
                        column_mapping[key] = variant
                        break
                if key in column_mapping:
                    break
        
        # If we don't have all required columns, return None
        if len(column_mapping) < 3:
            return None
        
        # Create a merged dataset
        merged_data = self.get_merged_data()
        if merged_data is None:
            return None
        
        # Calculate efficiency
        results = []
        for area in self.get_areas():
            area_data = merged_data[merged_data[column_mapping['area']] == area]
            
            # Calculate metrics
            total_shipped = area_data[column_mapping['quantity_sent']].sum() if column_mapping['quantity_sent'] in area_data.columns else 0
            total_sold = area_data[column_mapping['units_sold']].sum() if column_mapping['units_sold'] in area_data.columns else 0
            
            efficiency = (total_sold / total_shipped * 100) if total_shipped > 0 else 0
            overshipping = total_shipped - total_sold if total_shipped > total_sold else 0
            undershipping = total_sold - total_shipped if total_sold > total_shipped else 0
            
            results.append({
                'area': area,
                'total_shipped': total_shipped,
                'total_sold': total_sold,
                'efficiency_percentage': efficiency,
                'overshipping': overshipping,
                'undershipping': undershipping
            })
        
        return pd.DataFrame(results)
