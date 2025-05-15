import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.visualization import SupplyChainVisualizer

st.set_page_config(
    page_title="Area Performance Analysis",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if data is loaded
if 'data_processor' not in st.session_state or st.session_state.data_processor is None:
    st.warning("Please upload your data files on the main page before accessing this analysis.")
    st.stop()

# Initialize the visualizer
visualizer = SupplyChainVisualizer()

# Page title
st.title("Area Performance Analysis")
st.markdown("Detailed analysis of individual area performance metrics.")

# Get profitability and efficiency data
profitability = st.session_state.data_processor.calculate_profitability()
efficiency = st.session_state.data_processor.get_shipment_efficiency()

if profitability is None or efficiency is None:
    st.warning("Insufficient data to perform area analysis. Please check your data files.")
    st.stop()

# Area selection
areas = st.session_state.data_processor.get_areas()
selected_area = st.selectbox("Select an area to analyze:", options=areas)

# Filter data for selected area
area_profit_data = profitability[profitability['area'] == selected_area].iloc[0] if not profitability[profitability['area'] == selected_area].empty else None
area_efficiency_data = efficiency[efficiency['area'] == selected_area].iloc[0] if not efficiency[efficiency['area'] == selected_area].empty else None

# Display area image
st.image("https://pixabay.com/get/g9277bb3f69e4ba88783d9a159990e6a636a59b60c7404cfb46c2af986e4bdcd1b1d96a6c32886e6176436cb7e13eb0c0fc77409ca9954ed277a4d64afb22044e_1280.jpg", 
         caption="Area Performance Metrics")

# Area performance summary
st.header(f"Performance Summary: {selected_area}")

if area_profit_data is not None and area_efficiency_data is not None:
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", f"${area_profit_data['total_revenue']:,.2f}")
    
    with col2:
        st.metric("Total Cost", f"${area_profit_data['total_cost']:,.2f}")
    
    with col3:
        st.metric("Profit", f"${area_profit_data['profit']:,.2f}")
    
    with col4:
        st.metric("Profit Margin", f"{area_profit_data['profit_margin']:.2f}%")
    
    # Efficiency metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Units Shipped", f"{area_efficiency_data['total_shipped']:,.0f}")
    
    with col2:
        st.metric("Total Units Sold", f"{area_efficiency_data['total_sold']:,.0f}")
    
    with col3:
        st.metric("Shipment Efficiency", f"{area_efficiency_data['efficiency_percentage']:.2f}%")
    
    # Detailed analysis
    st.header("Detailed Analysis")
    
    # Tabs for different analyses
    tabs = st.tabs(["Profitability", "Shipment Efficiency", "Time Series"])
    
    with tabs[0]:
        st.subheader("Profitability Analysis")
        
        # Compare area profitability with others
        profitability['profit_display'] = profitability['profit'].abs()
        fig = px.bar(
            profitability,
            x='area',
            y='profit_display',
            title=f"Profit Comparison: {selected_area} vs. Other Areas",
            color='area',
            color_discrete_sequence=['lightgrey'] * len(profitability) + ['green']
        )
        # Highlight the selected area and color negative profits red
        fig.update_traces(
            marker_color=[
                'green' if area == selected_area and profitability.loc[profitability['area'] == area, 'profit'].iloc[0] >= 0 else
                'red' if profitability.loc[profitability['area'] == area, 'profit'].iloc[0] < 0 else
                'lightgrey' for area in profitability['area']
            ]
        )
        fig.update_layout(yaxis_title='Profit (absolute value)', yaxis_tickformat=',')
        st.plotly_chart(fig, use_container_width=True)
        
        # Profit breakdown
        st.subheader("Profit Breakdown")
        
        profit_breakdown = pd.DataFrame({
            'Category': ['Revenue', 'Cost', 'Profit'],
            'Amount': [
                area_profit_data['total_revenue'],
                area_profit_data['total_cost'],
                area_profit_data['profit']
            ]
        })
        
        fig = px.bar(
            profit_breakdown,
            x='Category',
            y='Amount',
            title=f"Revenue, Cost, and Profit for {selected_area}",
            color='Category',
            color_discrete_map={
                'Revenue': 'green',
                'Cost': 'red',
                'Profit': 'blue'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Profit ranking
        profit_rank = profitability.sort_values('profit', ascending=False).reset_index()
        area_rank = profit_rank[profit_rank['area'] == selected_area].index[0] + 1
        
        st.info(f"{selected_area} ranks #{area_rank} out of {len(profitability)} areas by profit.")
    
    with tabs[1]:
        st.subheader("Shipment Efficiency Analysis")
        
        # Compare area efficiency with others
        fig = px.bar(
            efficiency,
            x='area',
            y='efficiency_percentage',
            title=f"Shipment Efficiency Comparison: {selected_area} vs. Other Areas",
            color='area',
            color_discrete_sequence=['lightgrey'] * len(efficiency) + ['blue']
        )
        
        # Highlight the selected area
        fig.update_traces(
            marker_color=['blue' if area == selected_area else 'lightgrey' for area in efficiency['area']]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Shipment vs. Sales comparison
        shipment_data = pd.DataFrame({
            'Category': ['Units Shipped', 'Units Sold'],
            'Quantity': [
                area_efficiency_data['total_shipped'],
                area_efficiency_data['total_sold']
            ]
        })
        
        fig = px.bar(
            shipment_data,
            x='Category',
            y='Quantity',
            title=f"Units Shipped vs. Units Sold for {selected_area}",
            color='Category',
            color_discrete_map={
                'Units Shipped': 'orange',
                'Units Sold': 'purple'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Efficiency insights
        if area_efficiency_data['efficiency_percentage'] < 80:
            st.warning(f"The shipment efficiency for {selected_area} is below optimal levels. Consider reducing shipment quantities.")
        elif area_efficiency_data['efficiency_percentage'] > 100:
            st.warning(f"Sales exceed shipments in {selected_area}. This may indicate missed sales opportunities or inventory inaccuracies.")
        else:
            st.success(f"The shipment efficiency for {selected_area} is within optimal range.")
    
    with tabs[2]:
        st.subheader("Time Series Analysis")
        
        # Get time series data for the selected area
        time_series_data = st.session_state.data_processor.get_time_series_data(area=selected_area)
        
        if time_series_data is not None and len(time_series_data) > 0:
            # Find available metrics for time series
            metric_options = {
                'units_sold': 'Units Sold',
                'revenue': 'Revenue',
                'quantity_sent': 'Quantity Shipped'
            }
            
            available_metrics = []
            for m in metric_options.keys():
                if m in time_series_data.columns:
                    available_metrics.append(m)
            
            if available_metrics:
                selected_metric = st.selectbox(
                    "Select metric to analyze over time:",
                    options=available_metrics,
                    format_func=lambda x: metric_options.get(x, x)
                )
                
                time_series_fig = visualizer.create_time_series_chart(
                    time_series_data,
                    metric=selected_metric,
                    area=selected_area
                )
                
                if time_series_fig is not None:
                    st.plotly_chart(time_series_fig, use_container_width=True)
                    
                    # Calculate trend
                    if len(time_series_data) >= 3:  # Need at least 3 points for trend analysis
                        date_col = None
                        for col in ['date', 'shipment_date', 'sales_date']:
                            if col in time_series_data.columns:
                                date_col = col
                                break
                                
                        if date_col is not None:
                            # Group by date and calculate sum/average
                            grouped_data = time_series_data.groupby(date_col)[selected_metric].sum().reset_index()
                            grouped_data = grouped_data.sort_values(by=date_col)
                            
                            # Check trend by comparing first and last quarters
                            first_quarter = grouped_data[selected_metric].iloc[:len(grouped_data)//4].mean()
                            last_quarter = grouped_data[selected_metric].iloc[-len(grouped_data)//4:].mean()
                            
                            pct_change = ((last_quarter - first_quarter) / first_quarter * 100) if first_quarter > 0 else 0
                            
                            if pct_change > 10:
                                st.success(f"Positive trend: {selected_metric} increased by {pct_change:.2f}% from the first to the last quarter.")
                            elif pct_change < -10:
                                st.warning(f"Negative trend: {selected_metric} decreased by {abs(pct_change):.2f}% from the first to the last quarter.")
                            else:
                                st.info(f"Stable trend: {selected_metric} changed by only {pct_change:.2f}% from the first to the last quarter.")
            else:
                st.warning("No suitable metrics available for time series analysis.")
        else:
            st.warning(f"No time series data available for {selected_area}.")
else:
    st.error(f"No data available for {selected_area}. Please select another area.")

# Performance comparison with other areas
st.header("Performance Comparison with Other Areas")

# Select comparison metrics
comparison_metrics = st.multiselect(
    "Select metrics for comparison:",
    options=['profit', 'profit_margin', 'efficiency_percentage', 'total_revenue', 'total_cost'],
    default=['profit', 'efficiency_percentage']
)

if comparison_metrics:
    # Create radar chart data
    merged_data = pd.merge(profitability, efficiency, on='area', how='outer')
    
    # Normalize metrics for radar chart
    radar_data = merged_data.copy()
    for metric in comparison_metrics:
        if metric in radar_data.columns:
            min_val = radar_data[metric].min()
            max_val = radar_data[metric].max()
            if max_val > min_val:
                radar_data[metric] = (radar_data[metric] - min_val) / (max_val - min_val)
    
    # Get data for selected area and average of all areas
    selected_area_data = radar_data[radar_data['area'] == selected_area]
    
    # Calculate mean only on numeric columns to avoid TypeError
    numeric_cols = radar_data.select_dtypes(include=['number']).columns
    average_data = radar_data[numeric_cols].mean()
    
    # Add area column back for display purposes
    average_data = pd.Series(average_data)
    average_data['area'] = 'Average'
    
    # Create radar chart
    fig = go.Figure()
    
    # Get values for selected area
    selected_values = []
    for metric in comparison_metrics:
        if not selected_area_data.empty and metric in selected_area_data.columns:
            selected_values.append(selected_area_data[metric].iloc[0])
        else:
            selected_values.append(0)
    
    # Get values for average
    avg_values = []
    for metric in comparison_metrics:
        if metric in average_data.index:
            avg_values.append(average_data[metric])
        else:
            avg_values.append(0)
    
    fig.add_trace(go.Scatterpolar(
        r=selected_values,
        theta=comparison_metrics,
        fill='toself',
        name=selected_area
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=comparison_metrics,
        fill='toself',
        name='Average of All Areas'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=f"Performance Comparison: {selected_area} vs. Average"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Area recommendations
st.header("Recommendations")

# Simple recommendations based on metrics
if area_profit_data is not None and area_efficiency_data is not None:
    recommendations = []
    
    # Profitability recommendations
    if area_profit_data['profit'] < 0:
        recommendations.append("❌ **Critical Issue**: Area is operating at a loss. Consider reducing shipments or increasing prices.")
    elif area_profit_data['profit_margin'] < 10:
        recommendations.append("⚠️ **Profit Margin**: Low profit margin. Analyze costs and consider optimizing shipment quantities.")
    else:
        recommendations.append("✅ **Profit**: Good profit margin. Continue current strategy with possible expansion.")
    
    # Efficiency recommendations
    if area_efficiency_data['efficiency_percentage'] < 70:
        recommendations.append("⚠️ **Efficiency**: Low shipment efficiency. Reduce shipment quantities to avoid excess inventory.")
    elif area_efficiency_data['efficiency_percentage'] > 100:
        recommendations.append("⚠️ **Stockouts**: Sales exceed shipments. Consider increasing shipment quantities.")
    else:
        recommendations.append("✅ **Efficiency**: Good shipment efficiency. Maintain current shipment levels.")
    
    # Overall recommendation
    if area_profit_data['profit'] > 0 and area_efficiency_data['efficiency_percentage'] >= 80:
        overall = "✅ **Overall**: High-performing area. Consider increasing investment and expanding product range."
    elif area_profit_data['profit'] <= 0 and area_efficiency_data['efficiency_percentage'] < 70:
        overall = "❌ **Overall**: Underperforming area. Consider major strategy revision or market exit."
    else:
        overall = "⚠️ **Overall**: Mixed performance. Focus on improving the weaker metrics while maintaining strengths."
    
    # Display recommendations
    for rec in recommendations:
        st.markdown(rec)
    
    st.markdown("---")
    st.markdown(overall)
