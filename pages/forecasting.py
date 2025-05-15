import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from utils.visualization import SupplyChainVisualizer
from utils.ml_models import DemandForecaster

st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if data is loaded
if 'data_processor' not in st.session_state or st.session_state.data_processor is None:
    st.warning("Please upload your data files on the main page before accessing forecasting.")
    st.stop()

# Initialize the visualizer and forecaster
visualizer = SupplyChainVisualizer()
forecaster = DemandForecaster()

# Page title
st.title("Demand Forecasting")
st.markdown("Predict future sales and optimize inventory based on historical data.")

# Display forecasting image
st.image("https://pixabay.com/get/g39b5ca5dcb34847d58dc095519a047f55c2cf50e85d85b6e5d790bae5dd163ea368889457bb67c8300b0398611fe575597255239f4231102c4f1d0e54cf59af7_1280.jpg", 
         caption="Demand Forecasting", use_container_width=True)

# Area selection
areas = st.session_state.data_processor.get_areas()
selected_area = st.selectbox("Select an area for forecasting:", options=["All Areas"] + areas)

# Get time series data
area_for_forecasting = None if selected_area == "All Areas" else selected_area
time_series_data = st.session_state.data_processor.get_time_series_data(area=area_for_forecasting)

if time_series_data is None or len(time_series_data) == 0:
    st.warning("Insufficient time series data for forecasting. Please check your data files.")
    st.stop()

# Find suitable target column
target_options = []
for col in ['units_sold', 'sales_quantity', 'quantity_sold', 'revenue']:
    if col in time_series_data.columns:
        target_options.append(col)

if not target_options:
    st.warning("No suitable target column found for forecasting. Please check your data files.")
    st.stop()

# Forecasting options
st.header("Forecasting Options")

col1, col2, col3 = st.columns(3)

with col1:
    target_col = st.selectbox(
        "Select target metric to forecast:",
        options=target_options,
        format_func=lambda x: {
            'units_sold': 'Units Sold',
            'sales_quantity': 'Sales Quantity',
            'quantity_sold': 'Quantity Sold',
            'revenue': 'Revenue'
        }.get(x, x)
    )

with col2:
    model_type = st.selectbox(
        "Select forecast model:",
        options=['linear', 'random_forest'],
        format_func=lambda x: {
            'linear': 'Linear Regression',
            'random_forest': 'Random Forest'
        }.get(x, x)
    )

with col3:
    forecast_periods = st.slider("Forecast horizon (days):", min_value=7, max_value=90, value=30)

# Find date column
date_col = None
for col in ['date', 'shipment_date', 'sales_date']:
    if col in time_series_data.columns:
        date_col = col
        break

if date_col is None:
    st.warning("No date column found in the data. Cannot perform time series forecasting.")
    st.stop()

# Train the model and generate forecast
if st.button("Generate Forecast"):
    with st.spinner("Training model and generating forecast..."):
        # Train the model
        metrics = forecaster.train(time_series_data, target_col=target_col, model_type=model_type)
        
        if metrics is not None:
            # Display model metrics
            st.subheader("Model Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Squared Error", f"{metrics['mse']:.4f}")
            
            with col2:
                st.metric("Root Mean Squared Error", f"{metrics['rmse']:.4f}")
            
            with col3:
                st.metric("Mean Absolute Error", f"{metrics['mae']:.4f}")
            
            with col4:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
            
            # Generate forecast
            forecast = forecaster.forecast(time_series_data, forecast_periods=forecast_periods)
            
            if forecast is not None:
                # Display forecast chart
                st.subheader("Forecast Results")
                
                # Get historical data for visualization
                historical_data = time_series_data.sort_values(by=date_col)
                
                # Create forecast chart
                forecast_fig = visualizer.create_forecast_chart(
                    historical_data, 
                    forecast, 
                    date_col, 
                    target_col
                )
                
                if forecast_fig is not None:
                    st.plotly_chart(forecast_fig, use_container_width=True)
                
                # Display forecast table
                st.subheader("Forecast Data")
                
                # Format dates for display
                forecast_display = forecast.copy()
                forecast_display[date_col] = forecast_display[date_col].dt.strftime('%Y-%m-%d')
                
                st.dataframe(forecast_display)
                
                # Calculate forecast statistics
                forecast_mean = forecast['predicted'].mean()
                forecast_min = forecast['predicted'].min()
                forecast_max = forecast['predicted'].max()
                
                st.markdown(f"**Forecast Summary:**")
                st.markdown(f"- Average forecasted {target_col}: **{forecast_mean:.2f}**")
                st.markdown(f"- Minimum forecasted {target_col}: **{forecast_min:.2f}**")
                st.markdown(f"- Maximum forecasted {target_col}: **{forecast_max:.2f}**")
                
                # If forecasting for a specific area, show recommendations
                if area_for_forecasting is not None:
                    st.subheader("Recommendations")
                    
                    # Calculate recommended shipment quantity
                    avg_forecast = forecast['predicted'].mean()
                    safety_stock = avg_forecast * 0.2  # 20% safety stock
                    recommended_quantity = round(avg_forecast + safety_stock)
                    
                    st.markdown(f"Based on the forecast for **{area_for_forecasting}**, we recommend:")
                    st.markdown(f"- **Estimated monthly demand**: {avg_forecast:.2f} units")
                    st.markdown(f"- **Recommended safety stock**: {safety_stock:.2f} units")
                    st.markdown(f"- **Recommended monthly shipment**: **{recommended_quantity}** units")
                    
                    # Additional recommendations based on forecast trend
                    first_week = forecast['predicted'][:7].mean()
                    last_week = forecast['predicted'][-7:].mean()
                    trend_pct = ((last_week - first_week) / first_week * 100) if first_week > 0 else 0
                    
                    if trend_pct > 10:
                        st.markdown("⬆️ The forecast shows an **increasing trend**. Consider gradually increasing shipments throughout the period.")
                    elif trend_pct < -10:
                        st.markdown("⬇️ The forecast shows a **decreasing trend**. Consider reducing shipments toward the end of the period.")
                    else:
                        st.markdown("➡️ The forecast shows a **stable trend**. Maintain consistent shipment levels throughout the period.")
            else:
                st.error("Failed to generate forecast. Please try a different model or check your data.")
        else:
            st.error("Failed to train the model. Please try a different model or check your data.")

# Advanced forecasting options
with st.expander("Advanced Forecasting Options"):
    st.subheader("Seasonal Decomposition")
    
    if st.button("Perform Seasonal Decomposition"):
        try:
            import statsmodels.api as sm
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Check if we have enough data for decomposition
            if len(time_series_data) < 14:
                st.warning("Not enough data for seasonal decomposition. Need at least 14 data points.")
            else:
                # Group by date and calculate the mean for the target column
                grouped_data = time_series_data.groupby(date_col)[target_col].mean().reset_index()
                grouped_data = grouped_data.sort_values(by=date_col)
                
                # Set the date as index
                ts_data = grouped_data.set_index(date_col)[target_col]
                
                # Determine frequency if possible
                if len(ts_data) >= 30:
                    period = 7  # Weekly seasonality (adjust as needed)
                else:
                    period = len(ts_data) // 2
                
                # Perform decomposition
                decomposition = seasonal_decompose(ts_data, model='additive', period=period)
                
                # Plot components
                trend = decomposition.trend
                seasonal = decomposition.seasonal
                residual = decomposition.resid
                
                # Create figure with subplots
                fig = go.Figure()
                
                # Original data
                fig.add_trace(go.Scatter(
                    x=ts_data.index,
                    y=ts_data.values,
                    mode='lines',
                    name='Original'
                ))
                
                # Trend component
                fig.add_trace(go.Scatter(
                    x=trend.index,
                    y=trend.values,
                    mode='lines',
                    name='Trend',
                    line=dict(color='red')
                ))
                
                # Update layout
                fig.update_layout(
                    title='Time Series Decomposition - Trend',
                    xaxis_title='Date',
                    yaxis_title=target_col,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Seasonal component
                fig2 = go.Figure()
                
                fig2.add_trace(go.Scatter(
                    x=seasonal.index,
                    y=seasonal.values,
                    mode='lines',
                    name='Seasonal',
                    line=dict(color='green')
                ))
                
                fig2.update_layout(
                    title='Seasonal Component',
                    xaxis_title='Date',
                    yaxis_title='Seasonal Effect',
                    height=300
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Residual component
                fig3 = go.Figure()
                
                fig3.add_trace(go.Scatter(
                    x=residual.index,
                    y=residual.values,
                    mode='lines',
                    name='Residual',
                    line=dict(color='purple')
                ))
                
                fig3.update_layout(
                    title='Residual Component',
                    xaxis_title='Date',
                    yaxis_title='Residual',
                    height=300
                )
                
                st.plotly_chart(fig3, use_container_width=True)
                
                # Interpretation
                st.subheader("Interpretation")
                
                # Trend analysis
                trend_start = trend.dropna().iloc[0]
                trend_end = trend.dropna().iloc[-1]
                trend_change = ((trend_end - trend_start) / trend_start * 100) if trend_start != 0 else 0
                
                if trend_change > 5:
                    st.markdown(f"📈 **Increasing Trend**: The data shows an overall increasing trend of {trend_change:.2f}%.")
                elif trend_change < -5:
                    st.markdown(f"📉 **Decreasing Trend**: The data shows an overall decreasing trend of {abs(trend_change):.2f}%.")
                else:
                    st.markdown("📊 **Stable Trend**: The data shows a relatively stable trend.")
                
                # Seasonality analysis
                seasonal_magnitude = seasonal.max() - seasonal.min()
                seasonal_impact = (seasonal_magnitude / ts_data.mean() * 100)
                
                if seasonal_impact > 20:
                    st.markdown(f"🔄 **Strong Seasonality**: There is significant seasonal variation ({seasonal_impact:.2f}% of the mean).")
                elif seasonal_impact > 5:
                    st.markdown(f"🔄 **Moderate Seasonality**: There is moderate seasonal variation ({seasonal_impact:.2f}% of the mean).")
                else:
                    st.markdown("🔄 **Weak Seasonality**: There is minimal seasonal variation in the data.")
                
                # Residual analysis
                residual_magnitude = residual.std()
                residual_impact = (residual_magnitude / ts_data.mean() * 100)
                
                if residual_impact > 20:
                    st.markdown(f"⚠️ **High Variability**: The data contains high random variability ({residual_impact:.2f}% of the mean), making predictions less reliable.")
                elif residual_impact > 10:
                    st.markdown(f"⚠️ **Moderate Variability**: The data contains moderate random variability ({residual_impact:.2f}% of the mean).")
                else:
                    st.markdown(f"✅ **Low Variability**: The data contains low random variability ({residual_impact:.2f}% of the mean), making predictions more reliable.")
        
        except Exception as e:
            st.error(f"Error performing seasonal decomposition: {e}")
