import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.visualization import SupplyChainVisualizer
from utils.ml_models import DemandForecaster
from utils.recommendations import ShipmentRecommender

st.set_page_config(
    page_title="Shipment Recommendations",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if data is loaded
if 'data_processor' not in st.session_state or st.session_state.data_processor is None:
    st.warning("Please upload your data files on the main page before accessing recommendations.")
    st.stop()

# Initialize the visualizer, forecaster, and recommender
visualizer = SupplyChainVisualizer()
forecaster = DemandForecaster()
recommender = ShipmentRecommender(st.session_state.data_processor, forecaster)

# Page title
st.title("Shipment Recommendations")
st.markdown("Get AI-powered recommendations to optimize your supply chain.")

# Display recommendations image
st.image("https://pixabay.com/get/g35e1114be79b049ed0abcc84bc5860c9fc8f21c498d9ad7fc4e59bdbcf05b69ef67f379ce0fe0d92d7329e215a674a7a84f89560ea639bf0008ad6a5570b5ee2_1280.jpg", 
         caption="Shipment Optimization")

# Generate area recommendations
area_recommendations = recommender.generate_area_recommendations()

if area_recommendations is None:
    st.warning("Insufficient data to generate recommendations. Please check your data files.")
    st.stop()

# Display alerts
st.header("Alerts")
alerts = recommender.get_alerts()

if alerts:
    # Group alerts by severity
    high_alerts = [a for a in alerts if a['severity'] == 'high']
    medium_alerts = [a for a in alerts if a['severity'] == 'medium']
    low_alerts = [a for a in alerts if a['severity'] == 'low']
    
    # Display high severity alerts
    if high_alerts:
        st.error("Critical Issues")
        for alert in high_alerts:
            st.markdown(f"**{alert['area']}**: {alert['message']}")
    
    # Display medium severity alerts
    if medium_alerts:
        st.warning("Warnings")
        for alert in medium_alerts:
            st.markdown(f"**{alert['area']}**: {alert['message']}")
    
    # Display low severity alerts
    if low_alerts:
        st.info("Information")
        for alert in low_alerts:
            st.markdown(f"**{alert['area']}**: {alert['message']}")
else:
    st.success("No alerts detected. All areas are performing within acceptable parameters.")

# Area performance overview
st.header("Area Performance Overview")

# Create a scatter plot of profit vs efficiency
fig = px.scatter(
    area_recommendations,
    x='profit_margin',
    y='efficiency_percentage',
    size='total_revenue',
    color='recommendation_score',
    hover_name='area',
    color_continuous_scale='RdYlGn',
    title="Area Performance Matrix: Profit Margin vs. Efficiency",
    labels={
        'profit_margin': 'Profit Margin (%)',
        'efficiency_percentage': 'Shipment Efficiency (%)',
        'total_revenue': 'Total Revenue',
        'recommendation_score': 'Recommendation Score (1-10)'
    }
)

fig.update_layout(
    xaxis_title="Profit Margin (%)",
    yaxis_title="Shipment Efficiency (%)",
    coloraxis_colorbar=dict(title="Recommendation Score"),
    height=600
)

# Add quadrant lines
fig.add_vline(x=10, line_dash="dash", line_color="gray")  # Vertical line at 10% profit margin
fig.add_hline(y=80, line_dash="dash", line_color="gray")  # Horizontal line at 80% efficiency

# Add quadrant annotations
fig.add_annotation(x=5, y=40, text="Poor Performance", showarrow=False, font=dict(size=14, color="red"))
fig.add_annotation(x=5, y=90, text="Efficient but Low Margin", showarrow=False, font=dict(size=14, color="orange"))
fig.add_annotation(x=20, y=40, text="Profitable but Inefficient", showarrow=False, font=dict(size=14, color="orange"))
fig.add_annotation(x=20, y=90, text="High Performance", showarrow=False, font=dict(size=14, color="green"))

st.plotly_chart(fig, use_container_width=True)

# Area recommendations
st.header("Area-Specific Recommendations")

# Sort areas by recommendation score
sorted_areas = area_recommendations.sort_values('recommendation_score', ascending=False)

# Create tabs for top, middle, and bottom performers
top_areas = sorted_areas.head(min(5, len(sorted_areas)//3 + 1))
middle_areas = sorted_areas.iloc[len(sorted_areas)//3:2*len(sorted_areas)//3]
bottom_areas = sorted_areas.tail(min(5, len(sorted_areas)//3 + 1))

tabs = st.tabs(["Top Performers", "Average Performers", "Underperformers"])

with tabs[0]:
    st.subheader("Top Performing Areas")
    st.markdown("These areas have the highest recommendation scores. Consider increasing investment.")
    
    for _, area in top_areas.iterrows():
        with st.expander(f"{area['area']} (Score: {area['recommendation_score']:.1f})"):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"**Profit Margin:** {area['profit_margin']:.2f}%")
                st.markdown(f"**Total Revenue:** ${area['total_revenue']:,.2f}")
                st.markdown(f"**Total Profit:** ${area['profit']:,.2f}")
            
            with col2:
                st.markdown(f"**Shipment Efficiency:** {area['efficiency_percentage']:.2f}%")
                st.markdown(f"**Units Shipped:** {area['total_shipped']:,.0f}")
                st.markdown(f"**Units Sold:** {area['total_sold']:,.0f}")
            
            st.markdown("---")
            st.markdown(f"**Recommendation:** {area['recommendation']}")
            st.markdown(f"**Action:** {area['recommended_action']}")

with tabs[1]:
    st.subheader("Average Performing Areas")
    st.markdown("These areas have moderate recommendation scores. Focus on specific improvements.")
    
    for _, area in middle_areas.iterrows():
        with st.expander(f"{area['area']} (Score: {area['recommendation_score']:.1f})"):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"**Profit Margin:** {area['profit_margin']:.2f}%")
                st.markdown(f"**Total Revenue:** ${area['total_revenue']:,.2f}")
                st.markdown(f"**Total Profit:** ${area['profit']:,.2f}")
            
            with col2:
                st.markdown(f"**Shipment Efficiency:** {area['efficiency_percentage']:.2f}%")
                st.markdown(f"**Units Shipped:** {area['total_shipped']:,.0f}")
                st.markdown(f"**Units Sold:** {area['total_sold']:,.0f}")
            
            st.markdown("---")
            st.markdown(f"**Recommendation:** {area['recommendation']}")
            st.markdown(f"**Action:** {area['recommended_action']}")

with tabs[2]:
    st.subheader("Underperforming Areas")
    st.markdown("These areas have the lowest recommendation scores. Consider strategic changes.")
    
    for _, area in bottom_areas.iterrows():
        with st.expander(f"{area['area']} (Score: {area['recommendation_score']:.1f})"):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"**Profit Margin:** {area['profit_margin']:.2f}%")
                st.markdown(f"**Total Revenue:** ${area['total_revenue']:,.2f}")
                st.markdown(f"**Total Profit:** ${area['profit']:,.2f}")
            
            with col2:
                st.markdown(f"**Shipment Efficiency:** {area['efficiency_percentage']:.2f}%")
                st.markdown(f"**Units Shipped:** {area['total_shipped']:,.0f}")
                st.markdown(f"**Units Sold:** {area['total_sold']:,.0f}")
            
            st.markdown("---")
            st.markdown(f"**Recommendation:** {area['recommendation']}")
            st.markdown(f"**Action:** {area['recommended_action']}")

# Shipment quantity recommendations
st.header("Shipment Quantity Recommendations")

# Generate and display shipment quantity recommendations
if st.button("Generate Shipment Quantity Recommendations"):
    with st.spinner("Generating recommendations... This may take a moment."):
        # Generate recommendations for each area
        shipment_recs = recommender.generate_shipment_quantity_recommendations()
        
        if shipment_recs:
            # Convert to DataFrame for display
            rec_data = []
            for area, rec in shipment_recs.items():
                rec_data.append({
                    'Area': area,
                    'Forecast Avg Monthly Demand': rec['forecast_avg_monthly_demand'],
                    'Recommended Safety Stock': rec['recommended_safety_stock'],
                    'Recommended Monthly Shipment': rec['recommended_monthly_shipment']
                })
            
            rec_df = pd.DataFrame(rec_data)
            rec_df = rec_df.sort_values('Recommended Monthly Shipment', ascending=False)
            
            # Display recommendations
            st.subheader("Monthly Shipment Recommendations by Area")
            st.dataframe(rec_df)
            
            # Visualize recommendations
            fig = px.bar(
                rec_df,
                x='Area',
                y='Recommended Monthly Shipment',
                color='Forecast Avg Monthly Demand',
                color_continuous_scale='Viridis',
                title="Recommended Monthly Shipment Quantities by Area",
                labels={
                    'Area': 'Area',
                    'Recommended Monthly Shipment': 'Recommended Monthly Shipment',
                    'Forecast Avg Monthly Demand': 'Forecast Avg Demand'
                }
            )
            
            fig.update_layout(
                xaxis_title="Area",
                yaxis_title="Recommended Monthly Shipment",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient data to generate shipment quantity recommendations.")

# Area segmentation
st.header("Area Segmentation")
st.markdown("Group areas into segments based on performance characteristics.")

# Generate and display area segmentation
segments = recommender.segment_areas(n_clusters=3)

if segments is not None:
    # Display segmentation
    segment_counts = segments['segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Visualize segments
        fig = px.scatter(
            segments,
            x='profit_margin',
            y='efficiency_percentage',
            color='segment',
            size='total_revenue',
            hover_name='area',
            title="Area Segments: Profit Margin vs. Efficiency",
            labels={
                'profit_margin': 'Profit Margin (%)',
                'efficiency_percentage': 'Shipment Efficiency (%)',
                'total_revenue': 'Total Revenue',
                'segment': 'Segment'
            }
        )
        
        fig.update_layout(
            xaxis_title="Profit Margin (%)",
            yaxis_title="Shipment Efficiency (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Display segment counts
        st.subheader("Segment Distribution")
        
        fig = px.pie(
            segment_counts,
            values='Count',
            names='Segment',
            title="Distribution of Areas by Segment",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display segments with expandable details
    st.subheader("Areas by Segment")
    
    for segment in segments['segment'].unique():
        segment_areas = segments[segments['segment'] == segment]
        
        with st.expander(f"{segment} ({len(segment_areas)} areas)"):
            # Show segment characteristics
            avg_profit = segment_areas['profit'].mean()
            avg_margin = segment_areas['profit_margin'].mean()
            avg_efficiency = segment_areas['efficiency_percentage'].mean()
            
            st.markdown(f"**Average Profit:** ${avg_profit:,.2f}")
            st.markdown(f"**Average Profit Margin:** {avg_margin:.2f}%")
            st.markdown(f"**Average Shipment Efficiency:** {avg_efficiency:.2f}%")
            
            st.markdown("---")
            st.markdown("**Areas in this segment:**")
            
            # Display areas in a table
            st.dataframe(
                segment_areas[['area', 'profit', 'profit_margin', 'efficiency_percentage', 'total_revenue']]
                .sort_values('profit', ascending=False)
            )
else:
    st.warning("Insufficient data to perform area segmentation.")
