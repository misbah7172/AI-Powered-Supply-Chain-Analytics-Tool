import streamlit as st
st.set_page_config(page_title="Advanced Analytics", page_icon="📊", layout="wide")
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import utility modules
from utils.advanced_analytics import InventoryOptimizer, RouteOptimizer, RiskAssessor, WhatIfAnalyzer
from utils.export_tools import DataExporter, AlertManager
from utils.api_connector import APIConnector, get_connector_by_type

# Import ML modules with error handling
try:
    from utils.advanced_ml import AdvancedForecaster, AnomalyDetector, CustomerSegmentation, TENSORFLOW_AVAILABLE
except ImportError as e:
    st.error(f"Error importing ML modules: {str(e)}")
    # Define placeholder classes
    class AdvancedForecaster:
        def __init__(self):
            pass
        
    class AnomalyDetector:
        def __init__(self):
            pass
            
    class CustomerSegmentation:
        def __init__(self):
            pass
    
    TENSORFLOW_AVAILABLE = False

# Page configuration

# Check if data is loaded
if 'data_processor' not in st.session_state or st.session_state.data_processor is None:
    st.error("Please load data first on the main page.")
    st.stop()

# Get data from session state
shipment_data = st.session_state.shipment_data
sales_data = st.session_state.sales_data
inventory_data = st.session_state.inventory_data
cost_data = st.session_state.cost_data

# Page title and introduction
st.title("Advanced Supply Chain Analytics")
st.markdown("""
This page provides access to advanced analytics capabilities to optimize your supply chain operations.
Select a module from the sidebar to explore different analytics features.
""")

# Sidebar for module selection
with st.sidebar:
    st.header("Analytics Modules")
    module = st.radio(
        "Select Module:",
        ["Inventory Optimization", "Route Optimization", "Risk Assessment", 
         "What-If Analysis", "Advanced Forecasting", "Anomaly Detection",
         "Customer Segmentation", "Export & Reporting", "Alerts Dashboard"]
    )

# =========== INVENTORY OPTIMIZATION ===========
if module == "Inventory Optimization":
    st.header("Inventory Optimization")
    st.markdown("""
    Optimize inventory levels to minimize costs while maintaining service levels.
    Calculate economic order quantities (EOQ), safety stock levels, and reorder points.
    """)
    
    # Initialize the inventory optimizer
    inventory_optimizer = InventoryOptimizer(inventory_data, sales_data, cost_data)
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        areas = inventory_data['area'].unique()
        selected_area = st.selectbox("Select Area (optional)", options=["All Areas"] + list(areas))
    
    # Run optimization
    area_filter = None if selected_area == "All Areas" else selected_area
    
    with st.spinner("Calculating optimal inventory levels..."):
        optimization_results = inventory_optimizer.optimize_inventory_levels(area=area_filter)
    
    # Display results
    if not optimization_results.empty:
        st.subheader("Optimization Results")
        
        # Key Metrics Overview
        metrics_container = st.container()
        metrics_col1, metrics_col2, metrics_col3 = metrics_container.columns(3)
        
        # Calculate summary metrics
        total_cost = optimization_results['total_annual_inventory_cost'].sum()
        avg_turnover = optimization_results['inventory_turnover_ratio'].mean()
        avg_order_freq = optimization_results['order_frequency_per_year'].mean()
        
        metrics_col1.metric("Total Annual Inventory Cost", f"${total_cost:,.2f}")
        metrics_col2.metric("Average Inventory Turnover", f"{avg_turnover:.2f}x")
        metrics_col3.metric("Average Order Frequency", f"{avg_order_freq:.2f} orders/year")
        
        # Results DataTable
        with st.expander("Detailed Optimization Results", expanded=True):
            display_cols = ['area', 'product_category', 'annual_demand', 
                          'optimal_order_quantity', 'reorder_point', 'safety_stock',
                          'inventory_turnover_ratio', 'total_annual_inventory_cost']
            st.dataframe(optimization_results[display_cols])
        
        # Visualizations
        st.subheader("Inventory Optimization Visualizations")
        
        # Create visualizations
        figures = inventory_optimizer.create_inventory_optimization_visualization(optimization_results)
        
        # Display visualizations
        if 'costs' in figures:
            st.plotly_chart(figures['costs'], use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        if 'turnover' in figures:
            col1.plotly_chart(figures['turnover'], use_container_width=True)
        
        if 'eoq' in figures:
            col2.plotly_chart(figures['eoq'], use_container_width=True)
        
        if 'safety' in figures:
            st.plotly_chart(figures['safety'], use_container_width=True)
        
        # Download options
        exporter = DataExporter()
        with st.expander("Export Results"):
            export_format = st.radio("Export Format", ["CSV", "Excel", "PDF"])
            
            if export_format == "CSV":
                exporter.streamlit_download_button(
                    optimization_results, 'csv', 
                    label="Download CSV", 
                    filename_prefix="inventory_optimization"
                )
            elif export_format == "Excel":
                exporter.streamlit_download_button(
                    optimization_results, 'excel', 
                    label="Download Excel", 
                    filename_prefix="inventory_optimization"
                )
            elif export_format == "PDF":
                # Create PDF content
                pdf_content = f"""
                <h1>Inventory Optimization Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                
                <h2>Summary</h2>
                <ul>
                <li>Total Annual Inventory Cost: ${total_cost:,.2f}</li>
                <li>Average Inventory Turnover: {avg_turnover:.2f}x</li>
                <li>Average Order Frequency: {avg_order_freq:.2f} orders/year</li>
                </ul>
                
                <h2>Detailed Results</h2>
                {exporter.dataframe_to_html_table(optimization_results)}
                """
                
                exporter.streamlit_download_button(
                    pdf_content, 'pdf', 
                    label="Download PDF Report", 
                    filename_prefix="inventory_optimization"
                )
    else:
        st.info("Not enough data to perform optimization. Please check your data.")

# =========== ROUTE OPTIMIZATION ===========
elif module == "Route Optimization":
    st.header("Transportation Route Optimization")
    st.markdown("""
    Optimize delivery routes to minimize transportation costs and delivery times.
    Uses algorithms like the Traveling Salesman Problem (TSP) to find the optimal route.
    """)
    
    # Get locations from shipment data
    areas = shipment_data['area'].unique()
    
    # Create sample location data (since we don't have actual coordinates)
    # In a real app, you would use actual coordinates from your database
    location_data = pd.DataFrame({
        'area': areas,
        'latitude': np.random.uniform(30, 45, size=len(areas)),
        'longitude': np.random.uniform(-120, -70, size=len(areas))
    })
    
    # Initialize the route optimizer
    route_optimizer = RouteOptimizer(shipment_data, location_data)
    
    # Route optimization settings
    col1, col2 = st.columns(2)
    
    with col1:
        origin = st.selectbox("Select Origin", options=areas)
    
    with col2:
        algorithm = st.selectbox("Optimization Algorithm", 
                               options=["Traveling Salesman (TSP)", "Vehicle Routing Problem (VRP)", "Nearest Neighbor"])
        algorithm_map = {
            "Traveling Salesman (TSP)": "tsp",
            "Vehicle Routing Problem (VRP)": "vrp",
            "Nearest Neighbor": "nearest_neighbor"
        }
    
    # Options for selecting destinations
    include_all = st.checkbox("Include all areas as destinations", value=True)
    
    if not include_all:
        destinations = st.multiselect("Select Destinations", 
                                     options=[a for a in areas if a != origin],
                                     default=[a for a in areas if a != origin])
    else:
        destinations = [a for a in areas if a != origin]
    
    # Run optimization
    if st.button("Optimize Route"):
        if destinations:
            with st.spinner("Calculating optimal route..."):
                route_result = route_optimizer.optimize_route(
                    origin, 
                    destinations, 
                    algorithm=algorithm_map[algorithm]
                )
                
                # Display results
                st.subheader("Optimization Results")
                
                # Display route details
                if algorithm_map[algorithm] == "vrp":
                    # Multiple routes
                    st.write(f"Total Distance: {route_result['distance']:.2f} units")
                    st.write(f"Number of Routes: {len(route_result['routes'])}")
                    
                    for i, route in enumerate(route_result['routes']):
                        st.write(f"Route {i+1}: {' → '.join(route)}")
                else:
                    # Single route
                    st.write(f"Total Distance: {route_result['distance']:.2f} units")
                    st.write(f"Optimal Route: {' → '.join(route_result['route'])}")
                
                # Create and display route visualization
                route_map = route_optimizer.create_route_visualization(route_result)
                st.plotly_chart(route_map, use_container_width=True)
        else:
            st.error("Please select at least one destination.")

# =========== RISK ASSESSMENT ===========
elif module == "Risk Assessment":
    st.header("Supply Chain Risk Assessment")
    st.markdown("""
    Assess and visualize risks across your supply chain.
    Identify areas with high risk factors and get recommendations for risk mitigation.
    """)
    
    # Initialize the risk assessor
    risk_assessor = RiskAssessor(shipment_data, sales_data, inventory_data, cost_data)
    
    # Calculate risk scores
    with st.spinner("Calculating risk scores..."):
        risk_results = risk_assessor.calculate_risk_scores()
    
    # Display results
    if not risk_results.empty:
        st.subheader("Risk Assessment Results")
        
        # Key Metrics
        high_risk_areas = risk_results[risk_results['risk_category'] == 'High']['area'].tolist()
        avg_risk = risk_results['overall_risk_score'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Risk Score", f"{avg_risk:.2f}")
        col2.metric("High Risk Areas", len(high_risk_areas))
        col3.metric("Medium Risk Areas", len(risk_results[risk_results['risk_category'] == 'Medium']))
        
        # Results DataTable
        with st.expander("Detailed Risk Scores", expanded=True):
            display_cols = ['area', 'overall_risk_score', 'risk_category', 
                          'delivery_reliability_risk', 'stockout_risk', 
                          'demand_volatility_risk', 'cost_risk',
                          'supplier_concentration_risk', 'transportation_risk']
            
            st.dataframe(risk_results[display_cols])
        
        # If high risk areas exist, show them prominently
        if high_risk_areas:
            st.warning(f"High Risk Areas Detected: {', '.join(high_risk_areas)}")
        
        # Visualizations
        st.subheader("Risk Assessment Visualizations")
        
        # Create visualizations
        figures = risk_assessor.create_risk_visualization(risk_results)
        
        # Display visualizations
        if 'overall' in figures:
            st.plotly_chart(figures['overall'], use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        if 'radar' in figures:
            col1.plotly_chart(figures['radar'], use_container_width=True)
        
        if 'heatmap' in figures:
            col2.plotly_chart(figures['heatmap'], use_container_width=True)
        
        # Risk mitigation recommendations
        st.subheader("Risk Mitigation Recommendations")
        
        for idx, row in risk_results.iterrows():
            if row['risk_category'] == 'High':
                with st.expander(f"Recommendations for {row['area']} (High Risk)"):
                    st.write("### Key Risk Factors:")
                    risk_factors = []
                    
                    if row['delivery_reliability_risk'] > 0.6:
                        risk_factors.append("High delivery reliability risk")
                    if row['stockout_risk'] > 0.6:
                        risk_factors.append("High stockout risk")
                    if row['demand_volatility_risk'] > 0.6:
                        risk_factors.append("High demand volatility")
                    if row['cost_risk'] > 0.6:
                        risk_factors.append("High cost risk")
                    if row['supplier_concentration_risk'] > 0.6:
                        risk_factors.append("High supplier concentration")
                    if row['transportation_risk'] > 0.6:
                        risk_factors.append("High transportation risk")
                    
                    for factor in risk_factors:
                        st.write(f"- {factor}")
                    
                    st.write("### Recommended Actions:")
                    if row['delivery_reliability_risk'] > 0.6:
                        st.write("- Diversify carriers and transportation modes")
                        st.write("- Implement real-time shipment tracking")
                    if row['stockout_risk'] > 0.6:
                        st.write("- Increase safety stock levels")
                        st.write("- Implement improved forecasting methods")
                    if row['demand_volatility_risk'] > 0.6:
                        st.write("- Develop more responsive demand planning")
                        st.write("- Create contingency plans for demand surges")
                    if row['cost_risk'] > 0.6:
                        st.write("- Renegotiate contracts with suppliers")
                        st.write("- Optimize transportation routes to reduce costs")
            
            elif row['risk_category'] == 'Medium':
                with st.expander(f"Recommendations for {row['area']} (Medium Risk)"):
                    st.write("### Key Risk Factors:")
                    risk_factors = []
                    
                    if row['delivery_reliability_risk'] > 0.4:
                        risk_factors.append("Moderate delivery reliability risk")
                    if row['stockout_risk'] > 0.4:
                        risk_factors.append("Moderate stockout risk")
                    if row['demand_volatility_risk'] > 0.4:
                        risk_factors.append("Moderate demand volatility")
                    
                    for factor in risk_factors:
                        st.write(f"- {factor}")
                    
                    st.write("### Recommended Actions:")
                    st.write("- Monitor key risk indicators monthly")
                    st.write("- Develop contingency plans for moderate risk factors")
                    st.write("- Review inventory policies quarterly")
    else:
        st.info("Not enough data to perform risk assessment. Please check your data.")

# =========== WHAT-IF ANALYSIS ===========
elif module == "What-If Analysis":
    st.header("What-If Analysis")
    st.markdown("""
    Simulate different scenarios to understand their impact on your supply chain.
    Adjust parameters like demand, costs, lead times, and inventory levels to see the effects.
    """)
    
    # Initialize the what-if analyzer
    analyzer = WhatIfAnalyzer(shipment_data, sales_data, inventory_data, cost_data)
    
    # Calculate baseline metrics
    baseline_metrics = analyzer.calculate_scenario_metrics()
    
    # Create scenario selection
    st.subheader("Create Scenarios")
    
    scenario_type = st.selectbox(
        "Select Scenario Type",
        ["Demand Changes", "Cost Changes", "Lead Time Changes", "Inventory Level Changes"]
    )
    
    # Areas to modify
    areas = shipment_data['area'].unique()
    selected_area = st.selectbox("Select Area to Modify", options=["All Areas"] + list(areas))
    area_filter = None if selected_area == "All Areas" else selected_area
    
    # Scenario parameters based on type
    if scenario_type == "Demand Changes":
        # Demand change scenario
        st.write("##### Demand Change Parameters")
        
        if "product_category" in sales_data.columns:
            product_categories = sales_data['product_category'].unique()
            selected_product = st.selectbox(
                "Select Product Category (optional)", 
                options=["All Products"] + list(product_categories)
            )
            product_filter = None if selected_product == "All Products" else selected_product
        else:
            product_filter = None
        
        demand_change = st.slider(
            "Demand Change Percentage", 
            min_value=-50, 
            max_value=100, 
            value=20, 
            step=5,
            help="Positive values indicate increased demand, negative values indicate decreased demand"
        )
        
        # Create scenario
        if st.button("Run Demand Change Scenario"):
            # Reset to original data
            analyzer.reset_scenario()
            
            # Apply demand change
            analyzer.adjust_demand(area=area_filter, product_category=product_filter, change_percent=demand_change)
            
            # Calculate new metrics
            new_metrics = analyzer.calculate_scenario_metrics()
            
            # Compare scenarios
            scenarios = {
                "Baseline": baseline_metrics,
                f"Demand {demand_change:+d}%": new_metrics
            }
            
            comparison = analyzer.compare_scenarios(scenarios)
            
            # Display comparison
            st.subheader("Scenario Comparison")
            st.dataframe(comparison)
            
            # Visualizations
            st.subheader("Impact Analysis")
            
            metrics_to_viz = ['profit', 'profit_margin', 'total_revenue', 'total_costs']
            
            for metric in metrics_to_viz:
                fig = analyzer.create_scenario_visualization(scenarios, metric)
                st.plotly_chart(fig, use_container_width=True)
    
    elif scenario_type == "Cost Changes":
        # Cost change scenario
        st.write("##### Cost Change Parameters")
        
        cost_type = st.selectbox(
            "Select Cost Type to Modify",
            ["Transportation Costs", "Warehouse Costs", "Handling Costs", "Shipment Costs"]
        )
        
        cost_type_map = {
            "Transportation Costs": "transportation",
            "Warehouse Costs": "warehouse",
            "Handling Costs": "handling",
            "Shipment Costs": "shipment"
        }
        
        cost_change = st.slider(
            "Cost Change Percentage", 
            min_value=-30, 
            max_value=100, 
            value=15, 
            step=5,
            help="Positive values indicate increased costs, negative values indicate decreased costs"
        )
        
        # Create scenario
        if st.button("Run Cost Change Scenario"):
            # Reset to original data
            analyzer.reset_scenario()
            
            # Apply cost change
            analyzer.adjust_costs(cost_type_map[cost_type], area=area_filter, change_percent=cost_change)
            
            # Calculate new metrics
            new_metrics = analyzer.calculate_scenario_metrics()
            
            # Compare scenarios
            scenarios = {
                "Baseline": baseline_metrics,
                f"{cost_type} {cost_change:+d}%": new_metrics
            }
            
            comparison = analyzer.compare_scenarios(scenarios)
            
            # Display comparison
            st.subheader("Scenario Comparison")
            st.dataframe(comparison)
            
            # Visualizations
            st.subheader("Impact Analysis")
            
            metrics_to_viz = ['profit', 'profit_margin', 'total_costs']
            
            for metric in metrics_to_viz:
                fig = analyzer.create_scenario_visualization(scenarios, metric)
                st.plotly_chart(fig, use_container_width=True)
    
    elif scenario_type == "Lead Time Changes":
        # Lead time change scenario
        st.write("##### Lead Time Change Parameters")
        
        lead_time_change = st.slider(
            "Lead Time Change (Days)", 
            min_value=-7, 
            max_value=14, 
            value=3, 
            step=1,
            help="Positive values indicate increased lead times, negative values indicate decreased lead times"
        )
        
        # Create scenario
        if st.button("Run Lead Time Change Scenario"):
            # Reset to original data
            analyzer.reset_scenario()
            
            # Apply lead time change
            analyzer.adjust_lead_time(area=area_filter, change_days=lead_time_change)
            
            # Calculate new metrics
            new_metrics = analyzer.calculate_scenario_metrics()
            
            # Compare scenarios
            scenarios = {
                "Baseline": baseline_metrics,
                f"Lead Time {lead_time_change:+d} Days": new_metrics
            }
            
            comparison = analyzer.compare_scenarios(scenarios)
            
            # Display comparison
            st.subheader("Scenario Comparison")
            st.dataframe(comparison)
            
            # Visualizations
            st.subheader("Impact Analysis")
            
            metrics_to_viz = ['avg_delivery_time', 'profit']
            
            for metric in metrics_to_viz:
                fig = analyzer.create_scenario_visualization(scenarios, metric)
                st.plotly_chart(fig, use_container_width=True)
    
    elif scenario_type == "Inventory Level Changes":
        # Inventory level change scenario
        st.write("##### Inventory Level Change Parameters")
        
        if "product_category" in inventory_data.columns:
            product_categories = inventory_data['product_category'].unique()
            selected_product = st.selectbox(
                "Select Product Category (optional)", 
                options=["All Products"] + list(product_categories)
            )
            product_filter = None if selected_product == "All Products" else selected_product
        else:
            product_filter = None
        
        inventory_change = st.slider(
            "Inventory Level Change Percentage", 
            min_value=-50, 
            max_value=100, 
            value=30, 
            step=5,
            help="Positive values indicate increased inventory, negative values indicate decreased inventory"
        )
        
        # Create scenario
        if st.button("Run Inventory Level Change Scenario"):
            # Reset to original data
            analyzer.reset_scenario()
            
            # Apply inventory change
            analyzer.adjust_inventory(area=area_filter, product_category=product_filter, change_percent=inventory_change)
            
            # Calculate new metrics
            new_metrics = analyzer.calculate_scenario_metrics()
            
            # Compare scenarios
            scenarios = {
                "Baseline": baseline_metrics,
                f"Inventory {inventory_change:+d}%": new_metrics
            }
            
            comparison = analyzer.compare_scenarios(scenarios)
            
            # Display comparison
            st.subheader("Scenario Comparison")
            st.dataframe(comparison)
            
            # Visualizations
            st.subheader("Impact Analysis")
            
            metrics_to_viz = ['inventory_turnover', 'total_holding_cost', 'profit']
            
            for metric in metrics_to_viz:
                fig = analyzer.create_scenario_visualization(scenarios, metric)
                st.plotly_chart(fig, use_container_width=True)

# =========== ADVANCED FORECASTING ===========
elif module == "Advanced Forecasting":
    st.header("Advanced Demand Forecasting")
    st.markdown("""
    Forecasting with advanced models including Prophet and LSTM.
    Generate accurate predictions of future demand for better planning.
    """)
    
    # Initialize the forecaster
    forecaster = AdvancedForecaster()
    
    # Select forecasting method
    forecast_method = st.radio(
        "Select Forecasting Method",
        ["Prophet (Recommended)", "LSTM Neural Network"]
    )
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        if 'area' in sales_data.columns:
            areas = sales_data['area'].unique()
            selected_area = st.selectbox("Select Area", options=areas)
        else:
            selected_area = None
    
    with col2:
        if 'product_category' in sales_data.columns:
            products = sales_data['product_category'].unique()
            selected_product = st.selectbox("Select Product Category", options=products)
        else:
            selected_product = None
    
    # Forecast parameters
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_periods = st.slider("Forecast Periods (Days)", 30, 365, 90)
    
    with col2:
        target_col = st.selectbox(
            "Target Variable to Forecast", 
            options=["units_sold", "revenue"] if "revenue" in sales_data.columns else ["units_sold"]
        )
    
    # Filter data based on selection
    filtered_data = sales_data.copy()
    
    if selected_area is not None and 'area' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['area'] == selected_area]
    
    if selected_product is not None and 'product_category' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['product_category'] == selected_product]
    
    # Prepare time series data
    date_col = next((col for col in filtered_data.columns if 'date' in col.lower()), None)
    
    if date_col is None:
        st.error("No date column found in the data.")
        st.stop()
    
    # Ensure date column is datetime
    filtered_data[date_col] = pd.to_datetime(filtered_data[date_col])
    
    # Aggregate by date
    time_series_data = filtered_data.groupby(date_col)[target_col].sum().reset_index()
    
    # Show historical data
    st.subheader("Historical Data")
    st.line_chart(time_series_data.set_index(date_col))
    
    # Generate forecast
    if st.button("Generate Forecast"):
        with st.spinner(f"Generating forecast with {forecast_method}..."):
            if forecast_method.startswith("Prophet"):
                # Train Prophet model
                model_result = forecaster.train_prophet_model(
                    time_series_data,
                    date_col=date_col,
                    target_col=target_col
                )
                
                # Generate forecast
                forecast = forecaster.forecast_with_prophet(periods=forecast_periods)
                
                # Create forecast chart
                forecast_chart = forecaster.create_prophet_forecast_chart(forecast)
                
                # Display forecast
                st.subheader("Demand Forecast")
                st.plotly_chart(forecast_chart, use_container_width=True)
                
                # Display forecast components if available
                if 'weekly' in forecast.columns or 'yearly' in forecast.columns:
                    st.subheader("Forecast Components")
                    component_charts = forecaster.create_forecast_components_chart(forecast)
                    
                    for component, chart in component_charts.items():
                        st.plotly_chart(chart, use_container_width=True)
                
                # Display metrics
                st.subheader("Model Performance Metrics")
                metrics = model_result['metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean Squared Error (MSE)", f"{metrics['mse']:.2f}")
                col2.metric("Root Mean Squared Error (RMSE)", f"{metrics['rmse']:.2f}")
                col3.metric("Mean Absolute Error (MAE)", f"{metrics['mae']:.2f}")
                col4.metric("Mean Absolute Percentage Error (MAPE)", f"{metrics['mape']:.2f}%" if not np.isnan(metrics['mape']) else "N/A")
                
            else:  # LSTM
                # LSTM requires more data, check if we have enough
                if len(time_series_data) < 30:
                    st.error("LSTM requires at least 30 data points. Please select a different area/product with more data or use Prophet instead.")
                    st.stop()
                
                # Train LSTM model
                sequence_length = min(10, len(time_series_data) // 4)  # Adjust sequence length based on data size
                
                model_result = forecaster.train_lstm_model(
                    time_series_data,
                    date_col=date_col,
                    target_col=target_col,
                    sequence_length=sequence_length,
                    epochs=50
                )
                
                # Generate forecast
                forecast = forecaster.forecast_with_lstm(
                    time_series_data,
                    date_col=date_col,
                    target_col=target_col,
                    periods=forecast_periods
                )
                
                # Create forecast chart
                forecast_chart = forecaster.create_lstm_forecast_chart(
                    time_series_data,
                    date_col,
                    target_col,
                    forecast
                )
                
                # Display forecast
                st.subheader("Demand Forecast")
                st.plotly_chart(forecast_chart, use_container_width=True)
                
                # Display metrics
                st.subheader("Model Performance Metrics")
                metrics = model_result['metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean Squared Error (MSE)", f"{metrics['mse']:.2f}")
                col2.metric("Root Mean Squared Error (RMSE)", f"{metrics['rmse']:.2f}")
                col3.metric("Mean Absolute Error (MAE)", f"{metrics['mae']:.2f}")
                col4.metric("Mean Absolute Percentage Error (MAPE)", f"{metrics['mape']:.2f}%" if not np.isnan(metrics['mape']) else "N/A")
                
                # Display training loss curve if available
                if 'history' in model_result:
                    st.subheader("Training Progress")
                    history = model_result['history']
                    
                    if 'loss' in history and 'val_loss' in history:
                        loss_df = pd.DataFrame({
                            'epoch': range(1, len(history['loss']) + 1),
                            'training_loss': history['loss'],
                            'validation_loss': history['val_loss']
                        })
                        
                        loss_chart = px.line(
                            loss_df, 
                            x='epoch', 
                            y=['training_loss', 'validation_loss'],
                            title='Training and Validation Loss',
                            labels={'value': 'Loss', 'variable': 'Dataset'}
                        )
                        
                        st.plotly_chart(loss_chart, use_container_width=True)

# =========== ANOMALY DETECTION ===========
elif module == "Anomaly Detection":
    st.header("Anomaly Detection")
    st.markdown("""
    Detect unusual patterns and outliers in your supply chain data.
    Identify potential issues like unusual demand spikes, delivery delays, or cost anomalies.
    """)
    
    # Initialize the anomaly detector
    detector = AnomalyDetector()
    
    # Select data type
    data_type = st.selectbox(
        "Select Data Type",
        ["Sales Data", "Shipment Data", "Inventory Data", "Cost Data"]
    )
    
    # Map to actual dataframes
    data_map = {
        "Sales Data": sales_data,
        "Shipment Data": shipment_data,
        "Inventory Data": inventory_data,
        "Cost Data": cost_data
    }
    
    selected_data = data_map[data_type]
    
    # Select detection method
    method = st.radio(
        "Detection Method",
        ["Isolation Forest (Recommended)", "Z-Score"]
    )
    
    method_map = {
        "Isolation Forest (Recommended)": "isolation_forest",
        "Z-Score": "z_score"
    }
    
    # Select features
    numeric_cols = selected_data.select_dtypes(include=np.number).columns.tolist()
    
    if numeric_cols:
        selected_features = st.multiselect(
            "Select Features for Anomaly Detection",
            options=numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))]
        )
        
        # Additional parameters
        col1, col2 = st.columns(2)
        
        with col1:
            contamination = st.slider(
                "Expected Anomaly Percentage", 
                min_value=0.01, 
                max_value=0.2, 
                value=0.05,
                step=0.01,
                help="Percentage of data expected to be anomalies"
            )
        
        # Detect anomalies
        if st.button("Detect Anomalies") and selected_features:
            with st.spinner("Detecting anomalies..."):
                anomaly_results = detector.detect_anomalies(
                    selected_data,
                    method=method_map[method],
                    features=selected_features,
                    contamination=contamination
                )
                
                # Count anomalies
                anomaly_count = anomaly_results['anomaly'].sum()
                total_count = len(anomaly_results)
                anomaly_percent = (anomaly_count / total_count) * 100
                
                # Display results
                st.subheader("Anomaly Detection Results")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Records", total_count)
                col2.metric("Anomalies Found", anomaly_count)
                col3.metric("Anomaly Percentage", f"{anomaly_percent:.2f}%")
                
                # Show anomalies
                with st.expander("View Anomalies", expanded=True):
                    anomaly_only = anomaly_results[anomaly_results['anomaly']]
                    if not anomaly_only.empty:
                        st.dataframe(anomaly_only.drop(columns=['anomaly_score'] if 'anomaly_score' in anomaly_only.columns else []))
                    else:
                        st.info("No anomalies detected with the current settings.")
                
                # Visualizations
                st.subheader("Anomaly Visualizations")
                
                # First visualization: Scatter plot of two features
                if len(selected_features) >= 2:
                    scatter_chart = detector.create_anomaly_visualization(
                        anomaly_results,
                        x_col=selected_features[0],
                        y_col=selected_features[1],
                        label_col='area' if 'area' in anomaly_results.columns else None
                    )
                    
                    st.plotly_chart(scatter_chart, use_container_width=True)
                
                # Second visualization: Time series with anomalies
                if 'date' in anomaly_results.columns or any('date' in col.lower() for col in anomaly_results.columns):
                    date_col = next((col for col in anomaly_results.columns if 'date' in col.lower()), None)
                    
                    if date_col and selected_features:
                        time_series_chart = detector.create_time_series_anomaly_chart(
                            anomaly_results,
                            date_col=date_col,
                            value_col=selected_features[0]
                        )
                        
                        st.plotly_chart(time_series_chart, use_container_width=True)
        else:
            if not selected_features and st.button("Detect Anomalies"):
                st.warning("Please select at least one feature for anomaly detection.")
    else:
        st.error("No numeric columns found in the selected data.")

# =========== CUSTOMER SEGMENTATION ===========
elif module == "Customer Segmentation":
    st.header("Customer Segmentation")
    st.markdown("""
    Segment customers based on their behavior and characteristics.
    Identify different customer groups to optimize marketing and service strategies.
    """)
    
    # Check if we have the required data
    if 'customer_segment' not in sales_data.columns and 'customer_id' not in sales_data.columns:
        st.error("Customer segmentation requires sales data with customer identifiers. Please ensure your data has a 'customer_segment' or 'customer_id' column.")
        
        # Create dummy data for demonstration
        st.warning("Using synthetic customer data for demonstration purposes only.")
        
        # Generate synthetic customer IDs if not present
        demo_sales = sales_data.copy()
        
        if 'area' in demo_sales.columns:
            # Create customer IDs based on area and a random number
            areas = demo_sales['area'].unique()
            customer_ids = []
            
            for idx, row in demo_sales.iterrows():
                area_code = ''.join([word[0] for word in row['area'].split()]).upper()
                customer_id = f"CUST-{area_code}-{np.random.randint(100, 999)}"
                customer_ids.append(customer_id)
            
            demo_sales['customer_id'] = customer_ids
        else:
            # Create random customer IDs
            demo_sales['customer_id'] = [f"CUST-{i:04d}" for i in range(len(demo_sales))]
        
        # Use the demo data
        sales_for_segmentation = demo_sales
    else:
        # Use the actual data
        sales_for_segmentation = sales_data
    
    # Initialize the segmentation model
    segmenter = CustomerSegmentation()
    
    # Determine customer identifier column
    if 'customer_id' in sales_for_segmentation.columns:
        customer_col = 'customer_id'
    elif 'customer_segment' in sales_for_segmentation.columns:
        customer_col = 'customer_segment'
    else:
        customer_col = sales_for_segmentation.columns[0]  # Fallback to first column
    
    # Optional area and product columns
    area_col = 'area' if 'area' in sales_for_segmentation.columns else None
    product_col = 'product_category' if 'product_category' in sales_for_segmentation.columns else None
    
    # Prepare customer data
    with st.spinner("Preparing customer data..."):
        customer_data = segmenter.prepare_customer_data(
            sales_for_segmentation,
            customer_col=customer_col,
            area_col=area_col,
            product_col=product_col
        )
    
    if customer_data is not None and not customer_data.empty:
        # Segmentation parameters
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.slider("Number of Segments", min_value=2, max_value=6, value=3)
        
        with col2:
            features = st.multiselect(
                "Features for Segmentation",
                options=[col for col in customer_data.columns if col != 'customer'],
                default=[col for col in ['total_revenue', 'total_purchases', 'avg_purchase_value'] 
                         if col in customer_data.columns]
            )
        
        # Perform segmentation
        if st.button("Segment Customers") and features:
            with st.spinner("Segmenting customers..."):
                segmented_data = segmenter.segment_customers(
                    customer_data,
                    n_clusters=n_clusters,
                    features=features
                )
                
                # Analyze segments
                segment_profiles = segmenter.analyze_segments(segmented_data)
                
                # Display results
                st.subheader("Customer Segmentation Results")
                
                # Segment profiles
                st.write("##### Segment Profiles")
                profile_display = segment_profiles[['segment_label', 'customer_count', 'customer_percent']]
                
                for feature in features:
                    profile_display[f'avg_{feature}'] = segment_profiles[f'avg_{feature}']
                
                st.dataframe(profile_display)
                
                # Show customers by segment
                with st.expander("View Customers by Segment", expanded=True):
                    segment_tabs = st.tabs([f"Segment {i+1}" for i in range(n_clusters)])
                    
                    for i, tab in enumerate(segment_tabs):
                        segment_customers = segmented_data[segmented_data['segment'] == i]
                        with tab:
                            st.dataframe(segment_customers.drop(columns=['segment', 'pca_1', 'pca_2'] 
                                                              if 'pca_1' in segment_customers.columns else ['segment']))
                
                # Visualizations
                st.subheader("Segmentation Visualizations")
                
                # Create visualizations
                figures = segmenter.create_segment_visualization(segmented_data)
                
                # Display visualizations
                if 'scatter' in figures:
                    st.plotly_chart(figures['scatter'], use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                if 'radar' in figures:
                    col1.plotly_chart(figures['radar'], use_container_width=True)
                
                if 'bar' in figures:
                    col2.plotly_chart(figures['bar'], use_container_width=True)
                
                # Recommendations by segment
                st.subheader("Segment-based Recommendations")
                
                for i, row in segment_profiles.iterrows():
                    with st.expander(f"Recommendations for {row['segment_label']} Segment"):
                        segment_type = row['segment_label']
                        
                        if segment_type == 'High Value':
                            st.write("### Characteristics:")
                            st.write("- High average purchase value")
                            st.write("- Frequent purchases")
                            st.write("- High total revenue")
                            
                            st.write("### Recommendations:")
                            st.write("- Implement a loyalty program to reward and retain these customers")
                            st.write("- Provide premium service and personalized attention")
                            st.write("- Offer early access to new products or services")
                            st.write("- Develop cross-selling strategies for complementary products")
                        
                        elif segment_type == 'Medium Value':
                            st.write("### Characteristics:")
                            st.write("- Moderate purchase frequency and value")
                            st.write("- Potential for growth")
                            
                            st.write("### Recommendations:")
                            st.write("- Offer targeted promotions to increase purchase frequency")
                            st.write("- Implement a points system to encourage loyalty")
                            st.write("- Provide incentives for larger orders")
                            st.write("- Use email marketing to highlight product benefits")
                        
                        elif segment_type == 'Low Value':
                            st.write("### Characteristics:")
                            st.write("- Low purchase frequency and value")
                            st.write("- May be new or occasional customers")
                            
                            st.write("### Recommendations:")
                            st.write("- Send re-engagement emails with special offers")
                            st.write("- Implement a win-back campaign for inactive customers")
                            st.write("- Offer first-time purchase discounts")
                            st.write("- Simplify the purchase process to reduce barriers")
                        
                        else:
                            st.write("### Characteristics:")
                            for feature in features:
                                feature_value = row[f'avg_{feature}']
                                st.write(f"- Average {feature.replace('_', ' ')}: {feature_value:.2f}")
                            
                            st.write("### Recommendations:")
                            st.write("- Analyze this segment further to understand specific needs")
                            st.write("- Develop targeted marketing campaigns based on segment behavior")
                            st.write("- Test different engagement strategies to determine most effective approach")
        else:
            if not features and st.button("Segment Customers"):
                st.warning("Please select at least one feature for customer segmentation.")
    else:
        st.error("Could not prepare customer data. Please check your sales data format.")

# =========== EXPORT & REPORTING ===========
elif module == "Export & Reporting":
    st.header("Export & Reporting")
    st.markdown("""
    Generate reports and export data in various formats.
    Create shareable documents for stakeholders and team members.
    """)
    
    # Initialize the exporter
    exporter = DataExporter()
    
    # Select report type
    report_type = st.selectbox(
        "Select Report Type",
        ["Supply Chain Performance Overview", "Area Analysis Report", 
         "Inventory Status Report", "Shipment Efficiency Report",
         "Custom Data Export"]
    )
    
    if report_type == "Supply Chain Performance Overview":
        st.subheader("Supply Chain Performance Overview")
        
        # Key metrics
        metrics = {}
        
        # Sales metrics
        if 'revenue' in sales_data.columns:
            metrics['Total Revenue'] = sales_data['revenue'].sum()
        
        if 'units_sold' in sales_data.columns:
            metrics['Total Units Sold'] = sales_data['units_sold'].sum()
        
        # Cost metrics
        if 'shipment_cost' in shipment_data.columns:
            metrics['Total Shipment Cost'] = shipment_data['shipment_cost'].sum()
        
        if 'transportation_cost' in cost_data.columns:
            metrics['Total Transportation Cost'] = cost_data['transportation_cost'].sum()
        
        if 'warehouse_cost' in cost_data.columns:
            metrics['Total Warehouse Cost'] = cost_data['warehouse_cost'].sum()
        
        # Inventory metrics
        if 'remaining_stock' in inventory_data.columns:
            metrics['Total Inventory'] = inventory_data['remaining_stock'].sum()
        
        if 'holding_cost' in inventory_data.columns:
            metrics['Total Holding Cost'] = inventory_data['holding_cost'].sum()
        
        # Calculated metrics
        if 'revenue' in sales_data.columns and 'shipment_cost' in shipment_data.columns:
            profit = metrics.get('Total Revenue', 0) - metrics.get('Total Shipment Cost', 0)
            metrics['Estimated Profit'] = profit
            
            if metrics.get('Total Revenue', 0) > 0:
                metrics['Profit Margin'] = profit / metrics.get('Total Revenue', 0)
        
        # Figures and tables for the report
        figures = {}
        tables = {}
        
        # Create figures
        
        # Revenue by area
        if 'area' in sales_data.columns and 'revenue' in sales_data.columns:
            revenue_by_area = sales_data.groupby('area')['revenue'].sum().reset_index()
            fig_revenue = px.bar(
                revenue_by_area, 
                x='area', 
                y='revenue',
                title='Revenue by Area',
                labels={'revenue': 'Revenue', 'area': 'Area'}
            )
            figures['Revenue by Area'] = fig_revenue
        
        # Cost breakdown
        if 'transportation_cost' in cost_data.columns and 'warehouse_cost' in cost_data.columns:
            cost_data_sum = cost_data.sum()
            cost_breakdown = pd.DataFrame({
                'Cost Type': ['Transportation', 'Warehouse', 'Other'],
                'Amount': [
                    cost_data_sum.get('transportation_cost', 0),
                    cost_data_sum.get('warehouse_cost', 0),
                    cost_data_sum.get('handling_cost', 0) if 'handling_cost' in cost_data.columns else 0
                ]
            })
            fig_cost = px.pie(
                cost_breakdown,
                values='Amount',
                names='Cost Type',
                title='Cost Breakdown'
            )
            figures['Cost Breakdown'] = fig_cost
        
        # Shipment efficiency
        if 'delivery_time_days' in shipment_data.columns and 'area' in shipment_data.columns:
            delivery_time_by_area = shipment_data.groupby('area')['delivery_time_days'].mean().reset_index()
            fig_delivery = px.bar(
                delivery_time_by_area,
                x='area',
                y='delivery_time_days',
                title='Average Delivery Time by Area',
                labels={'delivery_time_days': 'Days', 'area': 'Area'}
            )
            figures['Delivery Time by Area'] = fig_delivery
        
        # Create tables
        
        # Top performing areas
        if 'area' in sales_data.columns and 'revenue' in sales_data.columns:
            top_areas = sales_data.groupby('area')['revenue'].sum().sort_values(ascending=False).reset_index()
            top_areas.columns = ['Area', 'Revenue']
            top_areas = top_areas.head(5)
            tables['Top 5 Performing Areas'] = top_areas
        
        # Generate the report
        report_html = exporter.create_dashboard_report(
            "Supply Chain Performance Overview",
            metrics,
            figures,
            tables
        )
        
        # Preview the report
        with st.expander("Preview Report", expanded=True):
            st.components.v1.html(report_html, height=600, scrolling=True)
        
        # Export options
        export_format = st.radio("Export Format", ["PDF", "Excel", "CSV", "JSON"])
        
        if export_format == "PDF":
            exporter.streamlit_download_button(
                report_html, 'pdf', 
                label="Download PDF Report", 
                filename_prefix="supply_chain_overview"
            )
        elif export_format == "Excel":
            # Create Excel workbook with multiple sheets
            excel_data = {}
            
            # Add metrics
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            excel_data['Overview'] = metrics_df
            
            # Add tables
            for name, table in tables.items():
                excel_data[name] = table
            
            # Add area data
            if 'area' in sales_data.columns:
                area_summary = sales_data.groupby('area').agg({
                    'revenue': 'sum' if 'revenue' in sales_data.columns else 'count',
                    'units_sold': 'sum' if 'units_sold' in sales_data.columns else 'count'
                }).reset_index()
                excel_data['Area Summary'] = area_summary
            
            exporter.streamlit_download_button(
                excel_data, 'excel', 
                label="Download Excel Report", 
                filename_prefix="supply_chain_overview"
            )
        elif export_format == "CSV":
            # Create a summary DataFrame
            summary_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            
            exporter.streamlit_download_button(
                summary_df, 'csv', 
                label="Download CSV Summary", 
                filename_prefix="supply_chain_overview"
            )
        elif export_format == "JSON":
            # Create a JSON object with all report data
            json_data = {
                'metrics': metrics,
                'tables': {name: table.to_dict('records') for name, table in tables.items()} if tables else {}
            }
            
            exporter.streamlit_download_button(
                json_data, 'json', 
                label="Download JSON Data", 
                filename_prefix="supply_chain_overview"
            )
    
    elif report_type == "Area Analysis Report":
        st.subheader("Area Analysis Report")
        
        # Select area
        areas = shipment_data['area'].unique()
        selected_area = st.selectbox("Select Area", options=areas)
        
        # Filter data for selected area
        area_shipments = shipment_data[shipment_data['area'] == selected_area]
        area_sales = sales_data[sales_data['area'] == selected_area]
        area_inventory = inventory_data[inventory_data['area'] == selected_area]
        area_costs = cost_data[cost_data['area'] == selected_area]
        
        # Key metrics
        metrics = {}
        
        # Sales metrics
        if 'revenue' in area_sales.columns:
            metrics['Total Revenue'] = area_sales['revenue'].sum()
        
        if 'units_sold' in area_sales.columns:
            metrics['Total Units Sold'] = area_sales['units_sold'].sum()
        
        # Cost metrics
        if 'shipment_cost' in area_shipments.columns:
            metrics['Total Shipment Cost'] = area_shipments['shipment_cost'].sum()
        
        if 'transportation_cost' in area_costs.columns:
            metrics['Total Transportation Cost'] = area_costs['transportation_cost'].sum()
        
        if 'warehouse_cost' in area_costs.columns:
            metrics['Total Warehouse Cost'] = area_costs['warehouse_cost'].sum()
        
        # Inventory metrics
        if 'remaining_stock' in area_inventory.columns:
            metrics['Total Inventory'] = area_inventory['remaining_stock'].sum()
        
        # Calculated metrics
        if 'revenue' in area_sales.columns and 'shipment_cost' in area_shipments.columns:
            profit = metrics.get('Total Revenue', 0) - metrics.get('Total Shipment Cost', 0)
            metrics['Estimated Profit'] = profit
            
            if metrics.get('Total Revenue', 0) > 0:
                metrics['Profit Margin'] = profit / metrics.get('Total Revenue', 0)
        
        # Figures and tables for the report
        figures = {}
        tables = {}
        
        # Create figures
        
        # Sales trend
        if 'sale_date' in area_sales.columns and 'revenue' in area_sales.columns:
            area_sales['sale_date'] = pd.to_datetime(area_sales['sale_date'])
            sales_trend = area_sales.groupby(area_sales['sale_date'].dt.to_period('M')).agg({
                'revenue': 'sum',
                'units_sold': 'sum' if 'units_sold' in area_sales.columns else 'count'
            }).reset_index()
            sales_trend['sale_date'] = sales_trend['sale_date'].dt.to_timestamp()
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=sales_trend['sale_date'],
                y=sales_trend['revenue'],
                mode='lines+markers',
                name='Revenue'
            ))
            
            fig_trend.update_layout(
                title=f'Sales Trend for {selected_area}',
                xaxis_title='Date',
                yaxis_title='Revenue'
            )
            
            figures['Sales Trend'] = fig_trend
        
        # Product breakdown
        if 'product_category' in area_sales.columns and 'revenue' in area_sales.columns:
            product_sales = area_sales.groupby('product_category')['revenue'].sum().reset_index()
            fig_products = px.pie(
                product_sales,
                values='revenue',
                names='product_category',
                title=f'Sales by Product Category in {selected_area}'
            )
            figures['Product Sales'] = fig_products
        
        # Inventory status
        if 'remaining_stock' in area_inventory.columns and 'product_category' in area_inventory.columns:
            inventory_status = area_inventory.groupby('product_category')['remaining_stock'].sum().reset_index()
            fig_inventory = px.bar(
                inventory_status,
                x='product_category',
                y='remaining_stock',
                title=f'Inventory Levels in {selected_area}',
                labels={'remaining_stock': 'Units in Stock', 'product_category': 'Product Category'}
            )
            figures['Inventory Status'] = fig_inventory
        
        # Create tables
        
        # Top selling products
        if 'product_category' in area_sales.columns and 'revenue' in area_sales.columns:
            top_products = area_sales.groupby('product_category').agg({
                'revenue': 'sum',
                'units_sold': 'sum' if 'units_sold' in area_sales.columns else 'count'
            }).sort_values('revenue', ascending=False).reset_index()
            top_products.columns = ['Product Category', 'Revenue', 'Units Sold']
            tables['Top Selling Products'] = top_products
        
        # Generate the report
        report_html = exporter.create_dashboard_report(
            f"Area Analysis Report: {selected_area}",
            metrics,
            figures,
            tables
        )
        
        # Preview the report
        with st.expander("Preview Report", expanded=True):
            st.components.v1.html(report_html, height=600, scrolling=True)
        
        # Export options
        export_format = st.radio("Export Format", ["PDF", "Excel", "CSV", "JSON"])
        
        if export_format == "PDF":
            exporter.streamlit_download_button(
                report_html, 'pdf', 
                label="Download PDF Report", 
                filename_prefix=f"area_analysis_{selected_area}"
            )
        elif export_format == "Excel":
            # Create Excel workbook with multiple sheets
            excel_data = {}
            
            # Add metrics
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            excel_data['Overview'] = metrics_df
            
            # Add tables
            for name, table in tables.items():
                excel_data[name] = table
            
            # Add raw data
            excel_data['Sales Data'] = area_sales
            excel_data['Inventory Data'] = area_inventory
            
            exporter.streamlit_download_button(
                excel_data, 'excel', 
                label="Download Excel Report", 
                filename_prefix=f"area_analysis_{selected_area}"
            )
        elif export_format == "CSV":
            # Create a summary DataFrame
            summary_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            
            exporter.streamlit_download_button(
                summary_df, 'csv', 
                label="Download CSV Summary", 
                filename_prefix=f"area_analysis_{selected_area}"
            )
        elif export_format == "JSON":
            # Create a JSON object with all report data
            json_data = {
                'area': selected_area,
                'metrics': metrics,
                'tables': {name: table.to_dict('records') for name, table in tables.items()} if tables else {}
            }
            
            exporter.streamlit_download_button(
                json_data, 'json', 
                label="Download JSON Data", 
                filename_prefix=f"area_analysis_{selected_area}"
            )
    
    elif report_type == "Custom Data Export":
        st.subheader("Custom Data Export")
        
        # Select data to export
        data_to_export = st.selectbox(
            "Select Data to Export",
            ["Sales Data", "Shipment Data", "Inventory Data", "Cost Data", "All Data"]
        )
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            if data_to_export != "All Data":
                areas = st.multiselect(
                    "Filter by Area (optional)",
                    options=shipment_data['area'].unique(),
                    default=[]
                )
            else:
                areas = []
        
        with col2:
            if data_to_export in ["Sales Data", "Inventory Data"] and "product_category" in sales_data.columns:
                products = st.multiselect(
                    "Filter by Product Category (optional)",
                    options=sales_data['product_category'].unique(),
                    default=[]
                )
            else:
                products = []
        
        # Prepare data based on selection
        export_data = {}
        
        if data_to_export == "Sales Data" or data_to_export == "All Data":
            filtered_sales = sales_data.copy()
            
            if areas:
                filtered_sales = filtered_sales[filtered_sales['area'].isin(areas)]
            
            if products and "product_category" in filtered_sales.columns:
                filtered_sales = filtered_sales[filtered_sales['product_category'].isin(products)]
            
            export_data["Sales Data"] = filtered_sales
        
        if data_to_export == "Shipment Data" or data_to_export == "All Data":
            filtered_shipments = shipment_data.copy()
            
            if areas:
                filtered_shipments = filtered_shipments[filtered_shipments['area'].isin(areas)]
            
            if products and "product_category" in filtered_shipments.columns:
                filtered_shipments = filtered_shipments[filtered_shipments['product_category'].isin(products)]
            
            export_data["Shipment Data"] = filtered_shipments
        
        if data_to_export == "Inventory Data" or data_to_export == "All Data":
            filtered_inventory = inventory_data.copy()
            
            if areas:
                filtered_inventory = filtered_inventory[filtered_inventory['area'].isin(areas)]
            
            if products and "product_category" in filtered_inventory.columns:
                filtered_inventory = filtered_inventory[filtered_inventory['product_category'].isin(products)]
            
            export_data["Inventory Data"] = filtered_inventory
        
        if data_to_export == "Cost Data" or data_to_export == "All Data":
            filtered_costs = cost_data.copy()
            
            if areas:
                filtered_costs = filtered_costs[filtered_costs['area'].isin(areas)]
            
            export_data["Cost Data"] = filtered_costs
        
        # Export options
        if export_data:
            export_format = st.radio("Export Format", ["Excel", "CSV", "JSON"])
            
            if st.button("Generate Export"):
                if export_format == "Excel":
                    exporter.streamlit_download_button(
                        export_data, 'excel', 
                        label="Download Excel Export", 
                        filename_prefix="supply_chain_data_export"
                    )
                elif export_format == "CSV":
                    if len(export_data) == 1:
                        # Single dataframe export
                        df_name = list(export_data.keys())[0]
                        df = list(export_data.values())[0]
                        
                        exporter.streamlit_download_button(
                            df, 'csv', 
                            label=f"Download {df_name} CSV", 
                            filename_prefix=df_name.lower().replace(" ", "_")
                        )
                    else:
                        st.error("CSV export is only available for a single data type. Please select a specific data type or use Excel for multiple datasets.")
                elif export_format == "JSON":
                    # Convert all dataframes to records
                    json_data = {
                        name: df.to_dict('records') for name, df in export_data.items()
                    }
                    
                    exporter.streamlit_download_button(
                        json_data, 'json', 
                        label="Download JSON Export", 
                        filename_prefix="supply_chain_data_export"
                    )

# =========== ALERTS DASHBOARD ===========
elif module == "Alerts Dashboard":
    st.header("Supply Chain Alerts Dashboard")
    st.markdown("""
    Monitor and manage alerts related to your supply chain operations.
    View warnings about potential issues like low inventory, delayed shipments, or demand surges.
    """)
    
    # Initialize the alert manager
    alert_mgr = AlertManager()
    
    # Get alerts
    with st.spinner("Analyzing data for alerts..."):
        alerts = alert_mgr.get_all_alerts(shipment_data, sales_data, inventory_data)
    
    # Display alerts
    if alerts:
        # Count alerts by severity
        high_alerts = len([a for a in alerts if a.get('severity') == 'high'])
        medium_alerts = len([a for a in alerts if a.get('severity') == 'medium'])
        low_alerts = len([a for a in alerts if a.get('severity') == 'low'])
        
        # Display alert counts
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Alerts", len(alerts))
        col2.metric("High Priority", high_alerts, delta=None, delta_color="inverse")
        col3.metric("Medium Priority", medium_alerts, delta=None, delta_color="off")
        col4.metric("Low Priority", low_alerts, delta=None, delta_color="off")
        
        # Display alerts by type
        st.subheader("Current Alerts")
        
        # Format alerts for display
        alert_df = alert_mgr.format_alerts_for_display(alerts)
        
        # Display alerts table
        st.dataframe(alert_df)
        
        # Group alerts by area
        st.subheader("Alerts by Area")
        
        area_tabs = st.tabs(sorted(alert_df['Area'].unique()))
        
        for i, tab in enumerate(area_tabs):
            area = sorted(alert_df['Area'].unique())[i]
            area_alerts = alert_df[alert_df['Area'] == area]
            
            with tab:
                st.dataframe(area_alerts)
        
        # Export alerts
        with st.expander("Export Alerts"):
            export_format = st.radio("Export Format", ["CSV", "Excel", "JSON"])
            
            if export_format == "CSV":
                exporter = DataExporter()
                exporter.streamlit_download_button(
                    alert_df, 'csv', 
                    label="Download Alerts CSV", 
                    filename_prefix="supply_chain_alerts"
                )
            elif export_format == "Excel":
                exporter = DataExporter()
                exporter.streamlit_download_button(
                    alert_df, 'excel', 
                    label="Download Alerts Excel", 
                    filename_prefix="supply_chain_alerts"
                )
            elif export_format == "JSON":
                exporter = DataExporter()
                exporter.streamlit_download_button(
                    alerts, 'json', 
                    label="Download Alerts JSON", 
                    filename_prefix="supply_chain_alerts"
                )
    else:
        st.success("No alerts detected. Your supply chain is operating normally.")
        
        # Sample alerts (for demonstration only)
        st.subheader("Sample Alerts (Example Only)")
        
        sample_alerts = [
            {
                'type': 'inventory',
                'severity': 'high',
                'area': 'New York',
                'product_category': 'Electronics',
                'message': "Low inventory alert: New York has only 50 units of Electronics remaining (below 20% of average)"
            },
            {
                'type': 'shipment',
                'severity': 'medium',
                'area': 'Chicago',
                'message': "Delayed shipment alert: Shipment to Chicago is taking 12 days (average is 7.5 days)"
            },
            {
                'type': 'demand',
                'severity': 'medium',
                'area': 'Los Angeles',
                'product_category': 'Clothing',
                'message': "Demand surge alert: Los Angeles sold 587 units of Clothing (average is 320 units)"
            }
        ]
        
        # Format sample alerts
        sample_alert_df = pd.DataFrame([
            {
                'Type': alert['type'].title(),
                'Severity': alert['severity'].title(),
                'Area': alert['area'],
                'Message': alert['message']
            }
            for alert in sample_alerts
        ])
        
        st.dataframe(sample_alert_df)