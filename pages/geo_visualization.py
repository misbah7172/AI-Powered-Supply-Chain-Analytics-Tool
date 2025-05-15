import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from utils.visualization import SupplyChainVisualizer

st.set_page_config(
    page_title="Geographic Visualization",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if data is loaded
if 'data_processor' not in st.session_state or st.session_state.data_processor is None:
    st.warning("Please upload your data files on the main page before accessing geographic visualization.")
    st.stop()

# Initialize the visualizer
visualizer = SupplyChainVisualizer()

# Page title
st.title("Geographic Visualization")
st.markdown("Visualize your supply chain performance on an interactive map.")

# Display geo visualization image
st.image("https://pixabay.com/get/g09383d61dbcb4739707869cffa8852c0c9ddd54342cdf7333bc1086b683ffbd4fd2c6edcc9a105caa7d52134b5781cf16dd6c75fae9d7cf89d6fa2a7052ebe72_1280.jpg",
         caption="Geographic Visualization")

# Get profitability and efficiency data
profitability = st.session_state.data_processor.calculate_profitability()
efficiency = st.session_state.data_processor.get_shipment_efficiency()

if profitability is None or efficiency is None:
    st.warning("Insufficient data to perform geographic analysis. Please check your data files.")
    st.stop()

# Location data input
st.header("Area Location Setup")
st.markdown("""
To visualize your supply chain on a map, you need to provide location data for each area.
You can either upload a CSV file with area coordinates or manually input them below.
""")

# Option to upload location data
location_file = st.file_uploader("Upload Area Location Data (CSV)", type=["csv"])

if location_file:
    try:
        location_data = pd.read_csv(location_file)
        st.success("Location data uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading location data: {e}")
        location_data = None
else:
    # Manual entry of location data
    st.subheader("Manual Location Entry")
    
    # Get list of areas
    areas = st.session_state.data_processor.get_areas()
    
    # Create a dataframe to store location data
    location_data = pd.DataFrame(columns=['area', 'latitude', 'longitude'])
    
    # Default coordinates for demonstration
    default_coordinates = {
        'New York': (40.7128, -74.0060),
        'Los Angeles': (34.0522, -118.2437),
        'Chicago': (41.8781, -87.6298),
        'Houston': (29.7604, -95.3698),
        'Phoenix': (33.4484, -112.0740),
        'Philadelphia': (39.9526, -75.1652),
        'San Antonio': (29.4241, -98.4936),
        'San Diego': (32.7157, -117.1611),
        'Dallas': (32.7767, -96.7970),
        'San Jose': (37.3382, -121.8863),
        'Austin': (30.2672, -97.7431),
        'Jacksonville': (30.3322, -81.6557),
        'Fort Worth': (32.7555, -97.3308),
        'Columbus': (39.9612, -82.9988),
        'Charlotte': (35.2271, -80.8431)
    }
    
    # Generate random coordinates for areas not in default list
    np.random.seed(42)  # For reproducibility
    
    # Create a form for manual entry
    with st.form("location_form"):
        st.markdown("Enter coordinates for each area:")
        
        # Create a container for all areas
        location_entries = []
        
        for i, area in enumerate(areas):
            col1, col2, col3 = st.columns([3, 2, 2])
            
            with col1:
                st.text(f"{area}")
            
            # Check if area name matches any in default coordinates
            default_lat, default_lon = None, None
            for def_area, (def_lat, def_lon) in default_coordinates.items():
                if def_area.lower() in area.lower():
                    default_lat, default_lon = def_lat, def_lon
                    break
            
            # If no match, generate random coordinates (within continental US for demo)
            if default_lat is None:
                default_lat = np.random.uniform(25, 49)  # US latitude range
                default_lon = np.random.uniform(-125, -65)  # US longitude range
            
            with col2:
                lat = st.number_input(f"Latitude for {area}", value=default_lat, format="%.6f", key=f"lat_{i}")
            
            with col3:
                lon = st.number_input(f"Longitude for {area}", value=default_lon, format="%.6f", key=f"lon_{i}")
            
            location_entries.append({
                'area': area,
                'latitude': lat,
                'longitude': lon
            })
        
        submit_button = st.form_submit_button("Save Locations")
        
        if submit_button:
            location_data = pd.DataFrame(location_entries)
            st.success("Location data saved successfully!")

# Geographic visualization
if location_data is not None and 'area' in location_data.columns and 'latitude' in location_data.columns and 'longitude' in location_data.columns:
    st.header("Supply Chain Map")
    
    # Metric selection
    metric_options = {
        'profit': 'Profit',
        'total_revenue': 'Revenue',
        'profit_margin': 'Profit Margin (%)',
        'efficiency_percentage': 'Shipment Efficiency (%)'
    }
    
    selected_metric = st.selectbox(
        "Select metric to visualize on the map:",
        options=list(metric_options.keys()),
        format_func=lambda x: metric_options[x]
    )
    
    # Merge location data with performance data
    if selected_metric in profitability.columns:
        merged_data = pd.merge(location_data, profitability, on='area', how='inner')
    elif selected_metric in efficiency.columns:
        merged_data = pd.merge(location_data, efficiency, on='area', how='inner')
    else:
        st.error(f"Selected metric '{selected_metric}' not found in data.")
        st.stop()
    
    if len(merged_data) == 0:
        st.warning("No matching areas found between location data and performance data.")
        st.stop()
    
    # Create a folium map
    st.subheader(f"Geographic Distribution of {metric_options[selected_metric]}")
    
    # Calculate center of the map
    center_lat = merged_data['latitude'].mean()
    center_lon = merged_data['longitude'].mean()
    
    # Create a base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
    
    # Normalize metric for color scaling
    max_val = merged_data[selected_metric].max()
    min_val = merged_data[selected_metric].min()
    
    # Add markers for each area
    marker_cluster = MarkerCluster().add_to(m)
    
    for idx, row in merged_data.iterrows():
        # Normalize value between 0 and 1
        normalized_value = (row[selected_metric] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        
        # Choose color based on value (red for low, green for high)
        color = f'#{int(255 * (1 - normalized_value)):02x}{int(255 * normalized_value):02x}00'
        
        # Create popup content
        popup_content = f"""
        <b>Area:</b> {row['area']}<br>
        <b>{metric_options[selected_metric]}:</b> {row[selected_metric]:.2f}<br>
        """
        
        # Add additional metrics based on which dataframe we're using
        if selected_metric in profitability.columns:
            popup_content += f"""
            <b>Revenue:</b> ${row['total_revenue']:,.2f}<br>
            <b>Cost:</b> ${row['total_cost']:,.2f}<br>
            """
        elif selected_metric in efficiency.columns:
            popup_content += f"""
            <b>Units Shipped:</b> {row['total_shipped']:,.0f}<br>
            <b>Units Sold:</b> {row['total_sold']:,.0f}<br>
            """
        
        # Create marker
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=10 + 20 * normalized_value,  # Size based on value
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_content, max_width=300)
        ).add_to(marker_cluster)
    
    # Display the map
    folium_static(m, width=1000, height=600)
    
    # Add a heatmap view option
    if st.checkbox("Show Heatmap View"):
        st.subheader(f"Heatmap of {metric_options[selected_metric]}")
        
        import plotly.graph_objects as go
        
        # Create a heatmap using Plotly
        fig = go.Figure(go.Densitymapbox(
            lat=merged_data['latitude'],
            lon=merged_data['longitude'],
            z=merged_data[selected_metric],
            radius=30,
            colorscale='RdYlGn',
            colorbar=dict(title=metric_options[selected_metric])
        ))
        
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=dict(lat=center_lat, lon=center_lon),
                zoom=3
            ),
            margin={"r":0,"t":0,"l":0,"b":0},
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Regional performance comparison
    st.header("Regional Performance Comparison")
    
    # Define regions based on latitude and longitude
    def assign_region(row):
        lat, lon = row['latitude'], row['longitude']
        
        # Simple region definition for US (customize as needed)
        if lon < -115:
            return "West"
        elif lon < -100:
            return "Mountain"
        elif lon < -85:
            return "Central"
        elif lon < -75:
            return "East"
        else:
            return "Northeast"
    
    # Add region to merged data
    merged_data['region'] = merged_data.apply(assign_region, axis=1)
    
    # Calculate regional aggregates
    regional_data = merged_data.groupby('region').agg({
        selected_metric: 'mean',
        'area': 'count'
    }).reset_index()
    
    regional_data.rename(columns={'area': 'number_of_areas'}, inplace=True)
    
    # Create bar chart for regional comparison
    fig = px.bar(
        regional_data,
        x='region',
        y=selected_metric,
        color=selected_metric,
        color_continuous_scale='RdYlGn',
        title=f"Average {metric_options[selected_metric]} by Region",
        labels={
            'region': 'Region',
            selected_metric: metric_options[selected_metric],
            'number_of_areas': 'Number of Areas'
        },
        text='number_of_areas'
    )
    
    fig.update_traces(
        texttemplate='%{text} areas',
        textposition='outside'
    )
    
    fig.update_layout(
        xaxis_title="Region",
        yaxis_title=metric_options[selected_metric],
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Distance analysis
    st.header("Distance Impact Analysis")
    st.markdown("""
    Analyze how distance between areas affects performance metrics.
    This can help identify logistics optimization opportunities.
    """)
    
    # Calculate distances between areas
    from scipy.spatial.distance import pdist, squareform
    
    # Extract coordinates
    coords = merged_data[['latitude', 'longitude']].values
    
    # Calculate distance matrix in km (approximate using Euclidean distance)
    dist_matrix = squareform(pdist(coords, lambda u, v: np.sqrt(
        (u[0]-v[0])**2 + (u[1]-v[1])**2
    ) * 111))  # 1 degree is approximately 111 km
    
    # Create a distance DataFrame
    distance_df = pd.DataFrame(
        dist_matrix,
        index=merged_data['area'],
        columns=merged_data['area']
    )
    
    # Show average distance to other areas
    merged_data['avg_distance_to_others'] = [
        distance_df.loc[area].mean() for area in merged_data['area']
    ]
    
    # Create scatter plot of distance vs. selected metric
    fig = px.scatter(
        merged_data,
        x='avg_distance_to_others',
        y=selected_metric,
        color='region',
        hover_name='area',
        title=f"Impact of Distance on {metric_options[selected_metric]}",
        labels={
            'avg_distance_to_others': 'Average Distance to Other Areas (km)',
            selected_metric: metric_options[selected_metric],
            'region': 'Region'
        },
        trendline='ols'  # Add trend line
    )
    
    fig.update_layout(
        xaxis_title="Average Distance to Other Areas (km)",
        yaxis_title=metric_options[selected_metric],
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate correlation
    correlation = merged_data[['avg_distance_to_others', selected_metric]].corr().iloc[0, 1]
    
    if abs(correlation) > 0.5:
        st.info(f"There appears to be a {'strong positive' if correlation > 0 else 'strong negative'} correlation ({correlation:.2f}) between distance and {metric_options[selected_metric]}.")
    elif abs(correlation) > 0.3:
        st.info(f"There appears to be a {'moderate positive' if correlation > 0 else 'moderate negative'} correlation ({correlation:.2f}) between distance and {metric_options[selected_metric]}.")
    else:
        st.info(f"There is a weak correlation ({correlation:.2f}) between distance and {metric_options[selected_metric]}.")
    
    # Export options
    st.header("Export Data")
    
    if st.button("Export Analyzed Location Data"):
        # Create a downloadable CSV
        csv = merged_data.to_csv(index=False)
        
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="geographic_analysis.csv",
            mime="text/csv",
        )
else:
    st.info("Please provide location data to see the geographic visualization.")
