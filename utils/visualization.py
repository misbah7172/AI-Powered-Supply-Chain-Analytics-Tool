import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

class SupplyChainVisualizer:
    """
    Class for creating visualizations for supply chain analytics.
    Provides various chart types and visualizations for analyzing supply chain data.
    """
    
    def create_area_performance_bar_chart(self, data, metric='profit', top_n=None, ascending=False):
        """
        Create a bar chart showing area performance based on a specified metric.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing area performance data
        metric : str
            The metric to visualize (e.g., 'profit', 'revenue', 'cost')
        top_n : int, optional
            If provided, show only the top N areas
        ascending : bool, default=False
            Sort in ascending order if True, descending if False
            
        Returns:
        --------
        plotly.graph_objects.Figure
            A plotly Figure object with the bar chart
        """
        if data is None or len(data) == 0:
            return None
            
        # Sort data
        sorted_data = data.sort_values(by=metric, ascending=ascending)
        
        # Take top N if specified
        if top_n is not None and top_n < len(sorted_data):
            sorted_data = sorted_data.head(top_n)
        
        # Create bar chart
        fig = px.bar(
            sorted_data,
            x='area',
            y=metric,
            title=f"{'Top' if not ascending else 'Bottom'} Areas by {metric.replace('_', ' ').title()}",
            color=metric,
            color_continuous_scale='Viridis',
            labels={'area': 'Area', metric: metric.replace('_', ' ').title()}
        )
        
        fig.update_layout(
            xaxis_title="Area",
            yaxis_title=metric.replace('_', ' ').title(),
            coloraxis_showscale=True
        )
        
        return fig
    
    def create_profit_vs_cost_scatter(self, data):
        """
        Create a scatter plot showing profit vs. cost for each area.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing area performance data with profit and cost metrics
            
        Returns:
        --------
        plotly.graph_objects.Figure
            A plotly Figure object with the scatter plot
        """
        if data is None or len(data) == 0:
            return None
            
        # Create scatter plot
        fig = px.scatter(
            data,
            x='total_cost',
            y='profit',
            text='area',
            size='total_revenue',
            color='profit_margin',
            color_continuous_scale='RdYlGn',
            title="Profit vs. Cost Analysis by Area",
            labels={
                'total_cost': 'Total Cost',
                'profit': 'Profit',
                'total_revenue': 'Total Revenue',
                'profit_margin': 'Profit Margin (%)'
            }
        )
        
        # Add a horizontal line at y=0 to indicate profit/loss threshold
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        
        fig.update_traces(
            textposition='top center',
            marker=dict(sizemode='diameter', sizeref=0.1)
        )
        
        fig.update_layout(
            xaxis_title="Total Cost",
            yaxis_title="Profit",
            coloraxis_colorbar=dict(title="Profit Margin (%)"),
            height=600
        )
        
        return fig
    
    def create_profit_heatmap(self, data):
        """
        Create a heatmap showing profit metrics across areas.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing area performance data
            
        Returns:
        --------
        plotly.graph_objects.Figure
            A plotly Figure object with the heatmap
        """
        if data is None or len(data) == 0:
            return None
            
        # Prepare data for heatmap
        metrics = ['total_revenue', 'total_cost', 'profit', 'profit_margin']
        heatmap_data = data.set_index('area')[metrics].T
        
        # Normalize data for better visualization
        normalized_data = heatmap_data.copy()
        for metric in metrics:
            if metric != 'profit_margin':  # Profit margin is already a percentage
                max_val = heatmap_data.loc[metric].max()
                min_val = heatmap_data.loc[metric].min()
                
                if max_val != min_val:
                    normalized_data.loc[metric] = (heatmap_data.loc[metric] - min_val) / (max_val - min_val)
        
        # Create heatmap
        fig = px.imshow(
            normalized_data,
            labels=dict(x="Area", y="Metric", color="Normalized Value"),
            x=normalized_data.columns,
            y=metrics,
            color_continuous_scale='RdYlGn',
            title="Profit Metrics Heatmap by Area",
            text_auto=True
        )
        
        fig.update_layout(
            height=500,
            xaxis=dict(tickangle=45)
        )
        
        return fig
    
    def create_shipment_efficiency_chart(self, data):
        """
        Create a chart comparing shipment efficiency across areas.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing shipment efficiency data
            
        Returns:
        --------
        plotly.graph_objects.Figure
            A plotly Figure object with the chart
        """
        if data is None or len(data) == 0:
            return None
            
        # Sort data by efficiency
        sorted_data = data.sort_values(by='efficiency_percentage', ascending=False)
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for shipped and sold quantities
        fig.add_trace(go.Bar(
            x=sorted_data['area'],
            y=sorted_data['total_shipped'],
            name='Total Shipped',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=sorted_data['area'],
            y=sorted_data['total_sold'],
            name='Total Sold',
            marker_color='darkblue'
        ))
        
        # Add line for efficiency percentage
        fig.add_trace(go.Scatter(
            x=sorted_data['area'],
            y=sorted_data['efficiency_percentage'],
            name='Efficiency (%)',
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ))
        
        # Update layout with second y-axis
        fig.update_layout(
            title="Shipment Efficiency by Area",
            xaxis_title="Area",
            yaxis_title="Quantity",
            yaxis2=dict(
                title="Efficiency (%)",
                overlaying="y",
                side="right",
                range=[0, 110]  # Assuming efficiency is a percentage
            ),
            barmode='group',
            legend=dict(x=0.01, y=0.99),
            height=600
        )
        
        return fig
    
    def create_time_series_chart(self, time_series_data, metric='units_sold', area=None):
        """
        Create a time series chart for sales/shipment data.
        
        Parameters:
        -----------
        time_series_data : pandas.DataFrame
            DataFrame containing time series data
        metric : str
            The metric to visualize
        area : str, optional
            If provided, highlight data for this specific area
            
        Returns:
        --------
        plotly.graph_objects.Figure
            A plotly Figure object with the time series chart
        """
        if time_series_data is None or len(time_series_data) == 0:
            return None
            
        # Find date column
        date_col = None
        for col in ['date', 'shipment_date', 'sales_date']:
            if col in time_series_data.columns:
                date_col = col
                break
                
        if date_col is None or metric not in time_series_data.columns:
            return None
            
        # Find area column
        area_col = None
        for col in ['area', 'area_name', 'region', 'location']:
            if col in time_series_data.columns:
                area_col = col
                break
                
        # Create time series chart
        if area_col is not None:
            # Group by date and area
            grouped_data = time_series_data.groupby([date_col, area_col])[metric].sum().reset_index()
            
            # Create figure
            fig = px.line(
                grouped_data, 
                x=date_col, 
                y=metric, 
                color=area_col,
                title=f"{metric.replace('_', ' ').title()} Over Time",
                labels={
                    date_col: "Date",
                    metric: metric.replace('_', ' ').title(),
                    area_col: "Area"
                }
            )
            
            # If an area is specified, highlight it
            if area is not None:
                for trace in fig.data:
                    if trace.name != area:
                        trace.line.color = "lightgrey"
                        trace.line.width = 1
                    else:
                        trace.line.width = 3
                        trace.line.color = "red"
        else:
            # Group by date only
            grouped_data = time_series_data.groupby(date_col)[metric].sum().reset_index()
            
            # Create figure
            fig = px.line(
                grouped_data, 
                x=date_col, 
                y=metric,
                title=f"{metric.replace('_', ' ').title()} Over Time",
                labels={
                    date_col: "Date",
                    metric: metric.replace('_', ' ').title()
                }
            )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=metric.replace('_', ' ').title(),
            height=500
        )
        
        return fig
    
    def create_forecast_chart(self, historical_data, forecast_data, date_col, metric):
        """
        Create a chart showing historical data and forecast.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            DataFrame containing historical time series data
        forecast_data : pandas.DataFrame
            DataFrame containing forecast data
        date_col : str
            Name of the date column
        metric : str
            The metric being forecasted
            
        Returns:
        --------
        plotly.graph_objects.Figure
            A plotly Figure object with the forecast chart
        """
        if historical_data is None or forecast_data is None:
            return None
            
        # Create figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_data[date_col],
            y=historical_data[metric],
            name="Historical Data",
            mode="lines",
            line=dict(color="blue")
        ))
        
        # Add forecast data
        fig.add_trace(go.Scatter(
            x=forecast_data[date_col],
            y=forecast_data['predicted'],
            name="Forecast",
            mode="lines",
            line=dict(color="red", dash="dash")
        ))
        
        # Add confidence interval if available
        if 'lower_bound' in forecast_data.columns and 'upper_bound' in forecast_data.columns:
            fig.add_trace(go.Scatter(
                x=forecast_data[date_col],
                y=forecast_data['upper_bound'],
                name="Upper Bound",
                mode="lines",
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_data[date_col],
                y=forecast_data['lower_bound'],
                name="Lower Bound",
                mode="lines",
                line=dict(width=0),
                fillcolor="rgba(255, 0, 0, 0.1)",
                fill='tonexty',
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title=f"{metric.replace('_', ' ').title()} Forecast",
            xaxis_title="Date",
            yaxis_title=metric.replace('_', ' ').title(),
            height=500,
            legend=dict(x=0.01, y=0.99)
        )
        
        return fig
    
    def create_geographic_heatmap(self, data, location_data, metric='profit'):
        """
        Create a geographic heatmap using Folium.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing area performance data
        location_data : pandas.DataFrame
            DataFrame with area names and coordinates (latitude, longitude)
        metric : str
            The metric to visualize
            
        Returns:
        --------
        folium.Map
            A Folium map object with the heatmap
        """
        import folium
        from folium.plugins import HeatMap
        
        if data is None or location_data is None:
            return None
            
        # Merge data with location data
        merged_data = pd.merge(
            data,
            location_data,
            on='area',
            how='inner'
        )
        
        if len(merged_data) == 0:
            return None
            
        # Create map centered at the mean location
        center_lat = merged_data['latitude'].mean()
        center_lon = merged_data['longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
        
        # Normalize metric for better visualization
        max_val = merged_data[metric].max()
        min_val = merged_data[metric].min()
        
        # Add markers for each area
        for idx, row in merged_data.iterrows():
            # Normalize value between 0 and 1
            normalized_value = (row[metric] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            
            # Choose color based on value (red for low, green for high)
            color = f'#{int(255 * (1 - normalized_value)):02x}{int(255 * normalized_value):02x}00'
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=10 + 20 * normalized_value,  # Size based on value
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"Area: {row['area']}<br>{metric}: {row[metric]:.2f}"
            ).add_to(m)
        
        return m
