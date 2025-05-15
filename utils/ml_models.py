import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class DemandForecaster:
    """
    Class for forecasting demand using different machine learning models.
    """
    
    def __init__(self):
        """Initialize the forecaster."""
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.features = []
        self.date_col = None
        self.target_col = None
        
    def _create_time_features(self, df, date_col):
        """
        Create time-based features from a date column.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the date column
        date_col : str
            Name of the date column
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional time features
        """
        # Create a copy to avoid modifying the original
        df_features = df.copy()
        
        # Extract time components
        df_features['year'] = df_features[date_col].dt.year
        df_features['month'] = df_features[date_col].dt.month
        df_features['quarter'] = df_features[date_col].dt.quarter
        df_features['week'] = df_features[date_col].dt.isocalendar().week
        df_features['day_of_week'] = df_features[date_col].dt.dayofweek
        df_features['day_of_month'] = df_features[date_col].dt.day
        df_features['day_of_year'] = df_features[date_col].dt.dayofyear
        
        # Create cyclical features for month, week, day
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        df_features['week_sin'] = np.sin(2 * np.pi * df_features['week'] / 52)
        df_features['week_cos'] = np.cos(2 * np.pi * df_features['week'] / 52)
        df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        
        # Drop any non-numeric columns that might cause issues
        # Only keep specified feature columns and exclude columns like 'revenue' 
        # that might be in the original data but not needed for forecasting
        
        return df_features
    
    def train(self, time_series_data, target_col='units_sold', model_type='linear'):
        """
        Train a forecasting model on the time series data.
        
        Parameters:
        -----------
        time_series_data : pandas.DataFrame
            DataFrame containing time series data
        target_col : str
            The target column to forecast
        model_type : str
            Type of model to use ('linear' or 'random_forest')
            
        Returns:
        --------
        dict
            Dictionary with model evaluation metrics
        """
        # Find date column
        date_col = None
        for col in ['date', 'shipment_date', 'sales_date']:
            if col in time_series_data.columns:
                date_col = col
                break
                
        if date_col is None or target_col not in time_series_data.columns:
            return None
            
        self.date_col = date_col
        self.target_col = target_col
        
        # Create features
        data = self._create_time_features(time_series_data, date_col)
        
        # Define features and target
        feature_cols = ['year', 'month', 'quarter', 'week', 'day_of_month', 
                        'month_sin', 'month_cos', 'week_sin', 'week_cos', 
                        'day_of_week_sin', 'day_of_week_cos']
        
        # Check if any additional numeric columns can be used as features
        for col in data.columns:
            if col != target_col and col != date_col and data[col].dtype in [np.int64, np.float64]:
                if col not in feature_cols and not col.startswith('day_of') and col != 'year':
                    feature_cols.append(col)
        
        self.features = feature_cols
        
        # Split into X and y
        X = data[feature_cols]
        y = data[target_col]
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # Scale target (for better model performance)
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        
        # Choose model
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            # Default to linear
            self.model = LinearRegression()
        
        # Train model
        self.model.fit(X_train_scaled, y_train_scaled)
        
        # Make predictions on test set
        y_test_scaled_pred = self.model.predict(X_test_scaled)
        y_test_pred = self.scaler_y.inverse_transform(y_test_scaled_pred.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def forecast(self, time_series_data, forecast_periods=30, confidence_interval=0.95):
        """
        Generate a forecast for future periods.
        
        Parameters:
        -----------
        time_series_data : pandas.DataFrame
            DataFrame containing time series data
        forecast_periods : int
            Number of periods to forecast
        confidence_interval : float
            Confidence interval for prediction bounds (0-1)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with forecasted values
        """
        if self.model is None or self.date_col is None or self.target_col is None:
            return None
            
        # Sort data by date
        time_series_data = time_series_data.sort_values(by=self.date_col)
        
        # Get the last date in the data
        last_date = time_series_data[self.date_col].max()
        
        # Create a dataframe for future dates
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_periods,
            freq='D'
        )
        
        future_df = pd.DataFrame({self.date_col: future_dates})
        
        # Create features for future dates
        future_df_features = self._create_time_features(future_df, self.date_col)
        
        # Check if all required features are available
        missing_features = [f for f in self.features if f not in future_df_features.columns]
        if missing_features:
            # If features are missing, add them with zeros
            for feature in missing_features:
                future_df_features[feature] = 0
                
        # Select only the features used by the model
        X_future = future_df_features[self.features]
        
        # Handle missing values
        X_future = X_future.fillna(X_future.mean())
        
        # Scale features
        X_future_scaled = self.scaler_X.transform(X_future)
        
        # Make predictions
        y_future_scaled = self.model.predict(X_future_scaled)
        y_future = self.scaler_y.inverse_transform(y_future_scaled.reshape(-1, 1)).flatten()
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            self.date_col: future_dates,
            'predicted': y_future
        })
        
        # Add confidence intervals if using a model that supports it
        if hasattr(self.model, 'predict_proba') or isinstance(self.model, RandomForestRegressor):
            # For RandomForest, we can use the standard deviation of predictions from different trees
            if isinstance(self.model, RandomForestRegressor):
                predictions = []
                for estimator in self.model.estimators_:
                    y_pred = estimator.predict(X_future_scaled)
                    predictions.append(y_pred)
                
                predictions = np.array(predictions)
                
                # Calculate confidence intervals
                std_dev = np.std(predictions, axis=0)
                z_value = 1.96  # 95% confidence interval
                
                # Scale back to original scale
                std_dev_original = std_dev * self.scaler_y.scale_
                
                forecast_df['lower_bound'] = forecast_df['predicted'] - z_value * std_dev_original
                forecast_df['upper_bound'] = forecast_df['predicted'] + z_value * std_dev_original
        else:
            # For models without built-in uncertainty, use a heuristic based on RMSE
            # Get the last part of the training data to estimate prediction error
            train_features = self._create_time_features(time_series_data, self.date_col)
            X_train = train_features[self.features].fillna(train_features[self.features].mean())
            X_train_scaled = self.scaler_X.transform(X_train)
            
            y_train = time_series_data[self.target_col]
            y_train_pred_scaled = self.model.predict(X_train_scaled)
            y_train_pred = self.scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            
            # Use RMSE for confidence interval
            z_value = 1.96  # 95% confidence interval
            forecast_df['lower_bound'] = forecast_df['predicted'] - z_value * rmse
            forecast_df['upper_bound'] = forecast_df['predicted'] + z_value * rmse
        
        # Ensure lower bound is not negative for quantities
        if 'lower_bound' in forecast_df.columns:
            forecast_df['lower_bound'] = forecast_df['lower_bound'].clip(lower=0)
        
        return forecast_df
