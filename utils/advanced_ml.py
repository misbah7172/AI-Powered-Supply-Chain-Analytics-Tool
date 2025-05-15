"""
Advanced Machine Learning module for the Supply Chain Analytics tool.
Implements Prophet and LSTM models for time series forecasting,
anomaly detection, and customer segmentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform

# TensorFlow import with fallback
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("TensorFlow is not installed. LSTM forecasting will not be available.")

class AdvancedForecaster:
    """
    Advanced forecasting class with Prophet and LSTM capabilities.
    """
    
    def __init__(self):
        """Initialize the forecaster."""
        self.prophet_model = None
        self.lstm_model = None
        self.scaler = None
    
    def prepare_prophet_data(self, df, date_col, target_col):
        """
        Prepare data for Prophet model.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing time series data
        date_col : str
            Name of the date column
        target_col : str
            Name of the target column to forecast
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame formatted for Prophet
        """
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Create Prophet DataFrame with required column names
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Sort by date
        prophet_df = prophet_df.sort_values('ds')
        
        return prophet_df
    
    def train_prophet_model(self, df, date_col, target_col, seasonality_mode='multiplicative', 
                           yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False):
        """
        Train a Prophet forecasting model.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing time series data
        date_col : str
            Name of the date column
        target_col : str
            Name of the target column to forecast
        seasonality_mode : str
            'multiplicative' or 'additive' seasonality
        yearly_seasonality : bool or int
            Whether to include yearly seasonality
        weekly_seasonality : bool or int
            Whether to include weekly seasonality
        daily_seasonality : bool or int
            Whether to include daily seasonality
            
        Returns:
        --------
        dict
            Dictionary with model and metrics
        """
        # Prepare data for Prophet
        prophet_df = self.prepare_prophet_data(df, date_col, target_col)
        
        # Initialize and train Prophet model
        self.prophet_model = Prophet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality="auto" if yearly_seasonality is True else yearly_seasonality,
            weekly_seasonality="auto" if weekly_seasonality is True else weekly_seasonality,
            daily_seasonality="auto" if daily_seasonality is True else daily_seasonality
        )
        
        # Add more seasonality if needed
        if 'month' in df.columns:
            self.prophet_model.add_regressor('month')
        
        # Fit the model
        self.prophet_model.fit(prophet_df)
        
        # Create future dataframe for validation (last 10% of data)
        split_idx = int(len(prophet_df) * 0.9)
        train_df = prophet_df.iloc[:split_idx]
        test_df = prophet_df.iloc[split_idx:]
        
        # Predict on validation data
        forecast = self.prophet_model.predict(test_df[['ds']])
        
        # Calculate error metrics
        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values
        
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if (y_true != 0).all() else np.nan
        
        # Return model and metrics
        return {
            'model': self.prophet_model,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            }
        }
    
    def forecast_with_prophet(self, periods=90, freq='D', include_history=True):
        """
        Generate forecast using trained Prophet model.
        
        Parameters:
        -----------
        periods : int
            Number of periods to forecast
        freq : str
            Frequency of forecast ('D' for daily, 'W' for weekly, 'M' for monthly)
        include_history : bool
            Whether to include historical data in forecast
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with forecast
        """
        if self.prophet_model is None:
            st.error("Prophet model not trained. Call train_prophet_model first.")
            return None
        
        # Create future dataframe
        future = self.prophet_model.make_future_dataframe(periods=periods, freq=freq, include_history=include_history)
        
        # Generate forecast
        forecast = self.prophet_model.predict(future)
        
        return forecast
    
    def prepare_lstm_data(self, df, date_col, target_col, sequence_length=10):
        """
        Prepare data for LSTM model.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing time series data
        date_col : str
            Name of the date column
        target_col : str
            Name of the target column to forecast
        sequence_length : int
            Number of time steps to use for each prediction
            
        Returns:
        --------
        tuple
            Tuple containing (X_train, y_train, X_test, y_test, scaler)
        """
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Extract target column and scale data
        data = df[target_col].values.reshape(-1, 1)
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X = []
        y = []
        
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i+sequence_length])
            y.append(scaled_data[i+sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and test sets (80/20)
        split_idx = int(len(X) * 0.8)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, y_train, X_test, y_test, self.scaler
    
    def train_lstm_model(self, df, date_col, target_col, sequence_length=10, epochs=50, batch_size=32):
        """
        Train an LSTM forecasting model.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing time series data
        date_col : str
            Name of the date column
        target_col : str
            Name of the target column to forecast
        sequence_length : int
            Number of time steps to use for each prediction
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
            
        Returns:
        --------
        dict
            Dictionary with model and metrics
        """
        # Check if TensorFlow is available
        if not TENSORFLOW_AVAILABLE:
            st.error("TensorFlow is not installed. Cannot train LSTM model.")
            return {
                'model': None,
                'scaler': None,
                'sequence_length': sequence_length,
                'metrics': {
                    'mse': np.nan,
                    'rmse': np.nan,
                    'mae': np.nan,
                    'mape': np.nan
                },
                'history': {'loss': [], 'val_loss': []}
            }
            
        # Prepare data for LSTM
        X_train, y_train, X_test, y_test, scaler = self.prepare_lstm_data(
            df, date_col, target_col, sequence_length
        )
        
        # Define LSTM model
        self.lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        # Compile model
        self.lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model with early stopping
        history = self.lstm_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Evaluate on test data
        y_pred = self.lstm_model.predict(X_test)
        
        # Invert scaling for actual values and predictions
        y_test_inv = scaler.inverse_transform(y_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        
        # Calculate error metrics
        mse = np.mean((y_test_inv - y_pred_inv) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_inv - y_pred_inv))
        mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100 if (y_test_inv != 0).all() else np.nan
        
        # Return model and metrics
        return {
            'model': self.lstm_model,
            'scaler': scaler,
            'sequence_length': sequence_length,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            },
            'history': history.history
        }
    
    def forecast_with_lstm(self, df, date_col, target_col, periods=30):
        """
        Generate forecast using trained LSTM model.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing historical time series data
        date_col : str
            Name of the date column
        target_col : str
            Name of the target column to forecast
        periods : int
            Number of periods to forecast
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with forecast
        """
        # Check if TensorFlow is available
        if not TENSORFLOW_AVAILABLE:
            st.error("TensorFlow is not installed. Cannot generate LSTM forecast.")
            # Create empty forecast DataFrame
            empty_forecast = pd.DataFrame({
                'ds': [(pd.to_datetime(df[date_col].iloc[-1]) if not df.empty else datetime.now()) + timedelta(days=i+1) for i in range(periods)],
                'yhat': np.zeros(periods),
                'yhat_lower': np.zeros(periods),
                'yhat_upper': np.zeros(periods)
            })
            return empty_forecast
            
        # Check if model is trained
        if self.lstm_model is None or self.scaler is None:
            st.error("LSTM model not trained. Call train_lstm_model first.")
            # Create empty forecast DataFrame
            empty_forecast = pd.DataFrame({
                'ds': [(pd.to_datetime(df[date_col].iloc[-1]) if not df.empty else datetime.now()) + timedelta(days=i+1) for i in range(periods)],
                'yhat': np.zeros(periods),
                'yhat_lower': np.zeros(periods),
                'yhat_upper': np.zeros(periods)
            })
            return empty_forecast
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Get the sequence length from the model's input shape
        sequence_length = self.lstm_model.input_shape[1]
        
        # Scale the data
        scaled_data = self.scaler.transform(df[target_col].values.reshape(-1, 1))
        
        # Get the last sequence to start the forecast
        last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
        
        # Generate forecast
        forecast_scaled = []
        current_sequence = last_sequence.copy()
        
        for _ in range(periods):
            # Predict the next value
            next_value = self.lstm_model.predict(current_sequence)[0]
            
            # Add to forecast
            forecast_scaled.append(next_value)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[:, 1:, :], 
                                         next_value.reshape(1, 1, 1), 
                                         axis=1)
        
        # Invert scaling
        forecast_values = self.scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))
        
        # Create dates for forecast
        last_date = pd.to_datetime(df[date_col].iloc[-1])
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': forecast_values.flatten(),
            'yhat_lower': forecast_values.flatten() * 0.9,  # Simple confidence interval
            'yhat_upper': forecast_values.flatten() * 1.1   # Simple confidence interval
        })
        
        return forecast_df
    
    def create_prophet_forecast_chart(self, forecast, uncertainty=True):
        """
        Create a Prophet forecast chart.
        
        Parameters:
        -----------
        forecast : pandas.DataFrame
            Prophet forecast DataFrame
        uncertainty : bool
            Whether to show uncertainty intervals
            
        Returns:
        --------
        plotly.graph_objects.Figure
            A Plotly figure with the forecast chart
        """
        fig = go.Figure()
        
        # Add historical points
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='blue')
        ))
        
        # Add uncertainty intervals
        if uncertainty and 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                mode='lines',
                name='Lower Bound',
                fill='tonexty',
                fillcolor='rgba(0, 0, 255, 0.2)',
                line=dict(width=0),
                showlegend=False
            ))
        
        # Add actual data points if available
        if 'y' in forecast.columns:
            mask = ~forecast['y'].isna()
            fig.add_trace(go.Scatter(
                x=forecast['ds'][mask],
                y=forecast['y'][mask],
                mode='markers',
                name='Actual',
                marker=dict(color='black', size=4)
            ))
        
        # Update layout
        fig.update_layout(
            title='Time Series Forecast with Prophet',
            xaxis_title='Date',
            yaxis_title='Value',
            height=500,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        return fig
    
    def create_forecast_components_chart(self, forecast_components):
        """
        Create a chart showing Prophet forecast components.
        
        Parameters:
        -----------
        forecast_components : pandas.DataFrame
            Prophet forecast components DataFrame
            
        Returns:
        --------
        dict
            Dictionary with component figures
        """
        figures = {}
        
        # Create trend component chart
        if 'trend' in forecast_components.columns:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=forecast_components['ds'],
                y=forecast_components['trend'],
                mode='lines',
                name='Trend',
                line=dict(color='blue')
            ))
            
            fig_trend.update_layout(
                title='Trend Component',
                xaxis_title='Date',
                yaxis_title='Trend',
                height=300
            )
            figures['trend'] = fig_trend
        
        # Create yearly seasonality chart
        if 'yearly' in forecast_components.columns:
            fig_yearly = go.Figure()
            fig_yearly.add_trace(go.Scatter(
                x=forecast_components['ds'],
                y=forecast_components['yearly'],
                mode='lines',
                name='Yearly Seasonality',
                line=dict(color='green')
            ))
            
            fig_yearly.update_layout(
                title='Yearly Seasonality Component',
                xaxis_title='Date',
                yaxis_title='Effect',
                height=300
            )
            figures['yearly'] = fig_yearly
        
        # Create weekly seasonality chart
        if 'weekly' in forecast_components.columns:
            fig_weekly = go.Figure()
            fig_weekly.add_trace(go.Scatter(
                x=forecast_components['ds'],
                y=forecast_components['weekly'],
                mode='lines',
                name='Weekly Seasonality',
                line=dict(color='orange')
            ))
            
            fig_weekly.update_layout(
                title='Weekly Seasonality Component',
                xaxis_title='Date',
                yaxis_title='Effect',
                height=300
            )
            figures['weekly'] = fig_weekly
        
        return figures
    
    def create_lstm_forecast_chart(self, historical_df, date_col, target_col, forecast_df):
        """
        Create an LSTM forecast chart.
        
        Parameters:
        -----------
        historical_df : pandas.DataFrame
            DataFrame with historical data
        date_col : str
            Name of the date column
        target_col : str
            Name of the target column
        forecast_df : pandas.DataFrame
            DataFrame with forecast data
            
        Returns:
        --------
        plotly.graph_objects.Figure
            A Plotly figure with the forecast chart
        """
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_df[date_col],
            y=historical_df[target_col],
            mode='lines+markers',
            name='Historical',
            line=dict(color='black')
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='blue')
        ))
        
        # Add uncertainty intervals
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_lower'],
            mode='lines',
            name='Lower Bound',
            fill='tonexty',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(width=0),
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title='Time Series Forecast with LSTM',
            xaxis_title='Date',
            yaxis_title='Value',
            height=500,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        return fig

class AnomalyDetector:
    """
    Class for detecting anomalies in supply chain data.
    """
    
    def __init__(self):
        """Initialize the anomaly detector."""
        self.model = None
    
    def detect_anomalies(self, df, method='isolation_forest', features=None, contamination=0.05):
        """
        Detect anomalies in the data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing data to check for anomalies
        method : str
            Method to use ('isolation_forest', 'lof', 'z_score')
        features : list, optional
            List of feature columns to use for anomaly detection
        contamination : float
            Expected proportion of anomalies
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with anomaly indicators
        """
        # Make a copy of the input DataFrame
        result_df = df.copy()
        
        # If no features specified, use all numeric columns
        if features is None:
            features = df.select_dtypes(include=np.number).columns.tolist()
        
        # Select only the specified features
        X = df[features].copy()
        
        # Fill NaN values with mean
        X = X.fillna(X.mean())
        
        # Apply the selected method
        if method == 'isolation_forest':
            # Train Isolation Forest model
            self.model = IsolationForest(contamination=float(contamination), random_state=42)
            # Predict anomalies (-1 for anomalies, 1 for normal)
            result_df['anomaly'] = self.model.fit_predict(X)
            # Convert to boolean (True for anomalies)
            result_df['anomaly'] = result_df['anomaly'] == -1
            
        elif method == 'z_score':
            # Calculate Z-scores for each feature
            z_scores = pd.DataFrame()
            for col in features:
                z_scores[f'{col}_zscore'] = (X[col] - X[col].mean()) / X[col].std()
            
            # Mark as anomaly if any feature has abs(z_score) > 3
            threshold = 3
            result_df['anomaly'] = False
            
            for col in z_scores.columns:
                result_df['anomaly'] = result_df['anomaly'] | (z_scores[col].abs() > threshold)
        
        # Add anomaly score if using isolation forest
        if method == 'isolation_forest':
            # Get anomaly scores (-ve means more anomalous)
            result_df['anomaly_score'] = -self.model.decision_function(X)
        
        return result_df
    
    def create_anomaly_visualization(self, df, x_col, y_col, label_col=None):
        """
        Create visualization of anomalies.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with anomaly indicators
        x_col : str
            Column to use for x-axis
        y_col : str
            Column to use for y-axis
        label_col : str, optional
            Column to use for point labels
            
        Returns:
        --------
        plotly.graph_objects.Figure
            A Plotly figure with anomaly visualization
        """
        # Create scatter plot
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color='anomaly',
            symbol='anomaly',
            color_discrete_map={True: 'red', False: 'blue'},
            symbol_map={True: 'circle', False: 'circle'},
            size='anomaly_score' if 'anomaly_score' in df.columns else None,
            hover_name=label_col if label_col else None,
            title=f'Anomaly Detection: {y_col} vs {x_col}'
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        return fig
    
    def create_time_series_anomaly_chart(self, df, date_col, value_col):
        """
        Create time series chart with anomalies highlighted.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with anomaly indicators
        date_col : str
            Column with dates
        value_col : str
            Column with values to plot
            
        Returns:
        --------
        plotly.graph_objects.Figure
            A Plotly figure with time series anomaly chart
        """
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Create figure
        fig = go.Figure()
        
        # Add normal points
        normal_points = df[~df['anomaly']]
        fig.add_trace(go.Scatter(
            x=normal_points[date_col],
            y=normal_points[value_col],
            mode='lines+markers',
            name='Normal',
            line=dict(color='blue'),
            marker=dict(size=6)
        ))
        
        # Add anomaly points
        anomaly_points = df[df['anomaly']]
        fig.add_trace(go.Scatter(
            x=anomaly_points[date_col],
            y=anomaly_points[value_col],
            mode='markers',
            name='Anomaly',
            marker=dict(
                color='red',
                size=10,
                symbol='circle',
                line=dict(width=2, color='red')
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Time Series Anomaly Detection: {value_col}',
            xaxis_title='Date',
            yaxis_title=value_col,
            height=500,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        return fig

class CustomerSegmentation:
    """
    Class for customer segmentation based on purchasing patterns.
    """
    
    def __init__(self):
        """Initialize the customer segmentation model."""
        self.model = None
        self.features = None
        self.scaler = None
        self.pca = None
    
    def prepare_customer_data(self, sales_data, customer_col, area_col=None, product_col=None):
        """
        Prepare customer data for segmentation.
        
        Parameters:
        -----------
        sales_data : pandas.DataFrame
            DataFrame containing sales data
        customer_col : str
            Column containing customer identifier
        area_col : str, optional
            Column containing area information
        product_col : str, optional
            Column containing product information
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with aggregated customer features
        """
        # Ensure we have a customer column
        if customer_col not in sales_data.columns:
            st.error(f"Customer column '{customer_col}' not found in data.")
            return None
        
        # Group by customer and calculate metrics
        customer_metrics = []
        
        for customer, data in sales_data.groupby(customer_col):
            metrics = {
                'customer': customer,
                'total_purchases': len(data),
                'total_units': data['units_sold'].sum() if 'units_sold' in data.columns else 0,
                'total_revenue': data['revenue'].sum() if 'revenue' in data.columns else 0
            }
            
            # Calculate average purchase value
            metrics['avg_purchase_value'] = metrics['total_revenue'] / metrics['total_purchases'] if metrics['total_purchases'] > 0 else 0
            
            # Calculate purchase frequency if we have date information
            if 'sale_date' in data.columns:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(data['sale_date']):
                    data['sale_date'] = pd.to_datetime(data['sale_date'])
                
                # Calculate days between first and last purchase
                days_active = (data['sale_date'].max() - data['sale_date'].min()).days
                metrics['purchase_frequency'] = metrics['total_purchases'] / (days_active + 1)  # avoid division by zero
                
                # Calculate days since last purchase
                last_purchase = data['sale_date'].max()
                current_date = pd.Timestamp.now()
                metrics['days_since_last_purchase'] = (current_date - last_purchase).days
            else:
                metrics['purchase_frequency'] = 0
                metrics['days_since_last_purchase'] = 0
            
            # Add area preference if area column exists
            if area_col and area_col in data.columns:
                area_counts = data[area_col].value_counts()
                metrics['preferred_area'] = area_counts.index[0] if len(area_counts) > 0 else None
                metrics['area_diversity'] = len(area_counts) / len(data)
            
            # Add product preference if product column exists
            if product_col and product_col in data.columns:
                product_counts = data[product_col].value_counts()
                metrics['preferred_product'] = product_counts.index[0] if len(product_counts) > 0 else None
                metrics['product_diversity'] = len(product_counts) / len(data)
            
            customer_metrics.append(metrics)
        
        # Convert to DataFrame
        customer_df = pd.DataFrame(customer_metrics)
        
        return customer_df
    
    def segment_customers(self, customer_df, n_clusters=3, features=None):
        """
        Segment customers using K-Means clustering.
        
        Parameters:
        -----------
        customer_df : pandas.DataFrame
            DataFrame with customer metrics
        n_clusters : int
            Number of clusters to create
        features : list, optional
            List of features to use for clustering
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with customer segments
        """
        # Define features if not provided
        if features is None:
            self.features = [
                'total_purchases', 'total_units', 'total_revenue', 
                'avg_purchase_value', 'purchase_frequency', 'days_since_last_purchase'
            ]
            # Filter to include only available features
            self.features = [f for f in self.features if f in customer_df.columns]
        else:
            self.features = features
        
        # Prepare feature matrix
        X = customer_df[self.features].copy()
        
        # Scale the features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction if there are many features
        if len(self.features) > 2:
            self.pca = PCA(n_components=min(len(self.features), 2))
            X_pca = self.pca.fit_transform(X_scaled)
            customer_df['pca_1'] = X_pca[:, 0]
            customer_df['pca_2'] = X_pca[:, 1] if X_pca.shape[1] > 1 else 0
        
        # Apply K-Means clustering
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        customer_df['segment'] = self.model.fit_predict(X_scaled)
        
        # Add segment labels
        segment_labels = {
            0: 'Low Value',
            1: 'Medium Value',
            2: 'High Value'
        }
        
        # If we have more than 3 segments, create generic labels
        if n_clusters > 3:
            for i in range(3, n_clusters):
                segment_labels[i] = f'Segment {i+1}'
        
        customer_df['segment_label'] = customer_df['segment'].map(segment_labels)
        
        return customer_df
    
    def analyze_segments(self, segmented_df):
        """
        Analyze the characteristics of each segment.
        
        Parameters:
        -----------
        segmented_df : pandas.DataFrame
            DataFrame with customer segments
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with segment profiles
        """
        # Group by segment and calculate metrics
        segment_profiles = []
        
        for segment, data in segmented_df.groupby('segment'):
            profile = {
                'segment': segment,
                'segment_label': data['segment_label'].iloc[0],
                'customer_count': len(data),
                'customer_percent': len(data) / len(segmented_df) * 100
            }
            
            # Calculate average metrics for each feature
            for feature in self.features:
                profile[f'avg_{feature}'] = data[feature].mean()
                profile[f'median_{feature}'] = data[feature].median()
            
            segment_profiles.append(profile)
        
        # Convert to DataFrame and sort by segment
        profile_df = pd.DataFrame(segment_profiles).sort_values('segment')
        
        return profile_df
    
    def create_segment_visualization(self, segmented_df):
        """
        Create visualization of customer segments.
        
        Parameters:
        -----------
        segmented_df : pandas.DataFrame
            DataFrame with customer segments
            
        Returns:
        --------
        dict
            Dictionary with segment visualizations
        """
        figures = {}
        
        # Check if we have PCA components
        has_pca = 'pca_1' in segmented_df.columns and 'pca_2' in segmented_df.columns
        
        # Create scatter plot of segments
        if has_pca:
            fig_scatter = px.scatter(
                segmented_df,
                x='pca_1',
                y='pca_2',
                color='segment_label',
                hover_name='customer',
                hover_data=self.features,
                title='Customer Segments',
                labels={'pca_1': 'PCA Component 1', 'pca_2': 'PCA Component 2'}
            )
        else:
            # Use the top 2 features if PCA is not available
            fig_scatter = px.scatter(
                segmented_df,
                x=self.features[0] if len(self.features) > 0 else 'total_revenue',
                y=self.features[1] if len(self.features) > 1 else 'total_purchases',
                color='segment_label',
                hover_name='customer',
                hover_data=self.features,
                title='Customer Segments'
            )
        
        figures['scatter'] = fig_scatter
        
        # Create radar chart comparing segments
        segment_profiles = self.analyze_segments(segmented_df)
        
        # Normalize the feature values for radar chart
        radar_features = [f'avg_{feature}' for feature in self.features]
        radar_df = segment_profiles[['segment_label'] + radar_features].copy()
        
        # Scale the features to 0-1 range
        for feature in radar_features:
            feature_min = radar_df[feature].min()
            feature_max = radar_df[feature].max()
            if feature_max > feature_min:
                radar_df[feature] = (radar_df[feature] - feature_min) / (feature_max - feature_min)
        
        # Create radar chart
        fig_radar = go.Figure()
        
        for _, row in radar_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[f] for f in radar_features],
                theta=[f.replace('avg_', '').replace('_', ' ').title() for f in radar_features],
                fill='toself',
                name=row['segment_label']
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title='Segment Profiles',
            showlegend=True
        )
        figures['radar'] = fig_radar
        
        # Create bar chart of segment sizes
        fig_bar = px.bar(
            segment_profiles,
            x='segment_label',
            y='customer_count',
            text='customer_count',
            color='segment_label',
            title='Segment Sizes',
            labels={'segment_label': 'Segment', 'customer_count': 'Number of Customers'}
        )
        figures['bar'] = fig_bar
        
        return figures