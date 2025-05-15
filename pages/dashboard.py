import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.visualization import SupplyChainVisualizer

st.set_page_config(
    page_title="Supply Chain Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if data is loaded
if 'data_processor' not in st.session_state or st.session_state.data_processor is None:
    st.warning("Please upload your data files on the main page before accessing the dashboard.")
    st.stop()

# Initialize the visualizer
visualizer = SupplyChainVisualizer()

# Dashboard title
st.title("Supply Chain Analytics Dashboard")
st.markdown("### Key Performance Metrics and Visualizations")

# Display dashboard layout image
st.image("https://pixabay.com/get/gebe8670d70a3c41b43d1bc4223d7ca6ce6462b299408f527a24b74b7ade0577194a3600d6a76f783d0633d0b954151fd75ed1871d0ffc4c9181cf8a06e20c1d7_1280.jpg", 
         caption="Supply Chain Analytics Dashboard")

# Key Performance Indicators (KPIs)
st.header("Key Metrics")

# Get profitability data
profitability = st.session_state.data_processor.calculate_profitability()
efficiency = st.session_state.data_processor.get_shipment_efficiency()

if profitability is not None and efficiency is not None:
    # Create KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = profitability['total_revenue'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with col2:
        total_profit = profitability['profit'].sum()
        st.metric("Total Profit", f"${total_profit:,.2f}")
    
    with col3:
        avg_profit_margin = profitability['profit_margin'].mean()
        st.metric("Avg. Profit Margin", f"{avg_profit_margin:.2f}%")
    
    with col4:
        avg_efficiency = efficiency['efficiency_percentage'].mean()
        st.metric("Avg. Shipment Efficiency", f"{avg_efficiency:.2f}%")

    # Top performing areas section
    st.header("Top Performing Areas")
    
    # Area metric selection
    metric_options = {
        'profit': 'Profit',
        'total_revenue': 'Revenue',
        'profit_margin': 'Profit Margin (%)',
        'efficiency_percentage': 'Shipment Efficiency (%)'
    }
    
    col1, col2 = st.columns([2, 2])
    
    with col1:
        selected_metric = st.selectbox(
            "Select metric to rank areas by:",
            options=list(metric_options.keys()),
            format_func=lambda x: metric_options[x]
        )
    
    with col2:
        top_n = st.slider("Number of areas to display:", min_value=3, max_value=10, value=5)
    
    # Display bar chart of top areas
    if selected_metric in profitability.columns:
        # Use profitability data
        fig = visualizer.create_area_performance_bar_chart(
            profitability,
            metric=selected_metric,
            top_n=top_n
        )
    elif selected_metric in efficiency.columns:
        # Use efficiency data
        fig = visualizer.create_area_performance_bar_chart(
            efficiency,
            metric=selected_metric,
            top_n=top_n
        )
    else:
        fig = None
    
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    
    # Profit vs Cost Analysis
    st.header("Profit vs. Cost Analysis")
    
    profit_cost_fig = visualizer.create_profit_vs_cost_scatter(profitability)
    if profit_cost_fig is not None:
        st.plotly_chart(profit_cost_fig, use_container_width=True)
    
    # Shipment Efficiency
    st.header("Shipment Efficiency Analysis")
    
    efficiency_fig = visualizer.create_shipment_efficiency_chart(efficiency)
    if efficiency_fig is not None:
        st.plotly_chart(efficiency_fig, use_container_width=True)
    
    # Time Series Analysis
    st.header("Sales Trends Over Time")
    
    # Get time series data
    areas = st.session_state.data_processor.get_areas()
    
    col1, col2 = st.columns([2, 2])
    
    with col1:
        selected_area = st.selectbox(
            "Select area for time series analysis:",
            options=["All Areas"] + areas
        )
    
    with col2:
        metric_options = {
            'units_sold': 'Units Sold',
            'revenue': 'Revenue',
            'quantity_sent': 'Quantity Shipped'
        }
        
        available_metrics = []
        for m in metric_options.keys():
            if m in st.session_state.sales_data.columns or m in st.session_state.shipment_data.columns:
                available_metrics.append(m)
        
        if available_metrics:
            selected_ts_metric = st.selectbox(
                "Select metric to analyze:",
                options=available_metrics,
                format_func=lambda x: metric_options.get(x, x)
            )
        else:
            selected_ts_metric = None
    
    if selected_ts_metric:
        # Get time series data for the selected area (or all areas)
        area_for_ts = None if selected_area == "All Areas" else selected_area
        time_series_data = st.session_state.data_processor.get_time_series_data(area=area_for_ts)
        
        if time_series_data is not None and selected_ts_metric in time_series_data.columns:
            time_series_fig = visualizer.create_time_series_chart(
                time_series_data,
                metric=selected_ts_metric,
                area=area_for_ts
            )
            
            if time_series_fig is not None:
                st.plotly_chart(time_series_fig, use_container_width=True)
    
    # Profit Metrics Heatmap
    st.header("Profit Metrics Heatmap by Area")
    
    heatmap_fig = visualizer.create_profit_heatmap(profitability)
    if heatmap_fig is not None:
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
else:
    st.warning("Insufficient data to generate performance metrics. Please check your data files.")
