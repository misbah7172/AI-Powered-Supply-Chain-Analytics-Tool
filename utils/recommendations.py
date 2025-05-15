import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class ShipmentRecommender:
    """
    Class for generating shipment recommendations based on historical data and forecasts.
    """
    
    def __init__(self, data_processor, forecaster):
        """
        Initialize the recommender with data processor and forecaster.
        
        Parameters:
        -----------
        data_processor : DataProcessor
            Instance of DataProcessor class with processed data
        forecaster : DemandForecaster
            Instance of DemandForecaster class with trained model
        """
        self.data_processor = data_processor
        self.forecaster = forecaster
    
    def generate_area_recommendations(self):
        """
        Generate recommendations for each area based on performance analysis.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with recommendations for each area
        """
        # Get profitability analysis
        profitability = self.data_processor.calculate_profitability()
        
        # Get shipment efficiency
        efficiency = self.data_processor.get_shipment_efficiency()
        
        if profitability is None or efficiency is None:
            return None
        
        # Merge profitability and efficiency data
        merged_data = pd.merge(profitability, efficiency, on='area', how='outer')
        
        # Calculate recommendation score (1-10 scale)
        # Higher score = higher priority for increased shipments
        scores = []
        recommendations = []
        actions = []
        
        for _, row in merged_data.iterrows():
            # Base components for scoring
            profit_component = 0
            efficiency_component = 0
            margin_component = 0
            
            # Profit component: higher profit = higher score
            if not pd.isna(row['profit']):
                profit_normalized = (row['profit'] - merged_data['profit'].min()) / (merged_data['profit'].max() - merged_data['profit'].min()) if (merged_data['profit'].max() - merged_data['profit'].min()) > 0 else 0.5
                profit_component = profit_normalized * 4  # 0-4 points
            
            # Efficiency component: higher efficiency = higher score
            if not pd.isna(row['efficiency_percentage']):
                efficiency_normalized = row['efficiency_percentage'] / 100  # Already 0-1
                efficiency_component = efficiency_normalized * 3  # 0-3 points
            
            # Margin component: higher margin = higher score
            if not pd.isna(row['profit_margin']):
                margin_normalized = (row['profit_margin'] - merged_data['profit_margin'].min()) / (merged_data['profit_margin'].max() - merged_data['profit_margin'].min()) if (merged_data['profit_margin'].max() - merged_data['profit_margin'].min()) > 0 else 0.5
                margin_component = margin_normalized * 3  # 0-3 points
            
            # Calculate total score
            score = profit_component + efficiency_component + margin_component
            score = min(max(score, 1), 10)  # Ensure score is between 1-10
            scores.append(score)
            
            # Generate recommendation text
            recommendation = ""
            action = ""
            
            if score >= 8:
                recommendation = "High-performing area with excellent profit and efficiency. Strong market potential."
                action = "Increase shipments significantly. Consider expanding product range or premium offerings."
            elif score >= 6:
                recommendation = "Good performance with positive profit margins and decent efficiency."
                action = "Moderately increase shipments. Monitor market response and adjust accordingly."
            elif score >= 4:
                recommendation = "Average performance. Some opportunities for improvement in efficiency or margins."
                action = "Maintain current shipment levels. Focus on optimizing logistics and reducing costs."
            elif score >= 2:
                recommendation = "Below average performance. Issues with efficiency or profitability need attention."
                action = "Slightly reduce shipments. Investigate causes of poor performance."
            else:
                recommendation = "Poor performance with significant efficiency or profitability issues."
                action = "Significantly reduce shipments. Consider restructuring distribution or exiting this market."
            
            recommendations.append(recommendation)
            actions.append(action)
        
        # Create result dataframe
        result = merged_data.copy()
        result['recommendation_score'] = scores
        result['recommendation'] = recommendations
        result['recommended_action'] = actions
        
        return result.sort_values(by='recommendation_score', ascending=False)
    
    def generate_shipment_quantity_recommendations(self, forecast_periods=30):
        """
        Generate recommendations for shipment quantities based on forecasts.
        
        Parameters:
        -----------
        forecast_periods : int
            Number of periods to forecast
            
        Returns:
        --------
        dict
            Dictionary with recommended shipment quantities for each area
        """
        # Get areas
        areas = self.data_processor.get_areas()
        
        recommendations = {}
        
        for area in areas:
            # Get time series data for the area
            time_series = self.data_processor.get_time_series_data(area=area)
            
            if time_series is None or len(time_series) == 0:
                continue
                
            # Find target column (units_sold or similar)
            target_col = None
            for col in ['units_sold', 'sales_quantity', 'quantity_sold']:
                if col in time_series.columns:
                    target_col = col
                    break
                    
            if target_col is None:
                continue
                
            # Train a model for this area
            self.forecaster.train(time_series, target_col=target_col)
            
            # Generate forecast
            forecast = self.forecaster.forecast(time_series, forecast_periods=forecast_periods)
            
            if forecast is None:
                continue
                
            # Calculate recommended shipment quantity
            # Base recommendation on forecast + safety stock
            
            # Calculate average forecast
            avg_forecast = forecast['predicted'].mean()
            
            # Add safety stock (20% of forecast)
            safety_stock = avg_forecast * 0.2
            
            # Calculate recommended monthly shipment
            recommended_quantity = avg_forecast + safety_stock
            
            # Round to nearest whole number
            recommended_quantity = round(recommended_quantity)
            
            # Store recommendation
            recommendations[area] = {
                'forecast_avg_monthly_demand': avg_forecast,
                'recommended_safety_stock': safety_stock,
                'recommended_monthly_shipment': recommended_quantity
            }
        
        return recommendations
    
    def segment_areas(self, n_clusters=3):
        """
        Segment areas based on performance metrics.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with area segments and characteristics
        """
        # Get profitability analysis
        profitability = self.data_processor.calculate_profitability()
        
        # Get shipment efficiency
        efficiency = self.data_processor.get_shipment_efficiency()
        
        if profitability is None or efficiency is None:
            return None
        
        # Merge profitability and efficiency data
        merged_data = pd.merge(profitability, efficiency, on='area', how='outer')
        
        # Select features for clustering
        features = ['profit', 'profit_margin', 'efficiency_percentage', 'total_revenue']
        
        # Handle missing values
        for feature in features:
            if feature in merged_data.columns:
                merged_data[feature] = merged_data[feature].fillna(merged_data[feature].median())
        
        # Extract features for clustering
        X = merged_data[features].values
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster information to data
        merged_data['cluster'] = clusters
        
        # Calculate cluster characteristics
        cluster_characteristics = merged_data.groupby('cluster')[features].mean().reset_index()
        
        # Label clusters
        if len(cluster_characteristics) == n_clusters:
            # Sort clusters by profit (or another key metric)
            cluster_characteristics = cluster_characteristics.sort_values(by='profit', ascending=False)
            
            # Rename clusters
            cluster_mapping = {}
            for i, index in enumerate(cluster_characteristics['cluster']):
                if i == 0:
                    cluster_mapping[index] = "High-Performing Areas"
                elif i == n_clusters - 1:
                    cluster_mapping[index] = "Underperforming Areas"
                else:
                    cluster_mapping[index] = f"Average Areas (Group {i})"
            
            # Map cluster names to original data
            merged_data['segment'] = merged_data['cluster'].map(cluster_mapping)
        
        # Return segmented data
        return merged_data[['area', 'cluster', 'segment', 'profit', 'profit_margin', 'efficiency_percentage', 'total_revenue']]
    
    def get_alerts(self):
        """
        Generate alerts for areas that need attention.
        
        Returns:
        --------
        list
            List of alert dictionaries with area and alert message
        """
        # Get profitability analysis
        profitability = self.data_processor.calculate_profitability()
        
        # Get shipment efficiency
        efficiency = self.data_processor.get_shipment_efficiency()
        
        if profitability is None or efficiency is None:
            return []
        
        alerts = []
        
        # Check for areas with negative profit
        negative_profit_areas = profitability[profitability['profit'] < 0]
        for _, row in negative_profit_areas.iterrows():
            alerts.append({
                'area': row['area'],
                'alert_type': 'profit',
                'severity': 'high',
                'message': f"Negative profit of {row['profit']:.2f} in {row['area']}. Immediate action required."
            })
        
        # Check for areas with low profit margin
        low_margin_areas = profitability[(profitability['profit'] > 0) & (profitability['profit_margin'] < 10)]
        for _, row in low_margin_areas.iterrows():
            alerts.append({
                'area': row['area'],
                'alert_type': 'margin',
                'severity': 'medium',
                'message': f"Low profit margin of {row['profit_margin']:.2f}% in {row['area']}. Cost optimization recommended."
            })
        
        # Check for areas with low shipment efficiency
        low_efficiency_areas = efficiency[efficiency['efficiency_percentage'] < 70]
        for _, row in low_efficiency_areas.iterrows():
            alerts.append({
                'area': row['area'],
                'alert_type': 'efficiency',
                'severity': 'medium',
                'message': f"Low shipment efficiency of {row['efficiency_percentage']:.2f}% in {row['area']}. Shipment quantities may need adjustment."
            })
        
        # Check for areas with high overshipping
        overshipping_areas = efficiency[efficiency['overshipping'] > efficiency['total_shipped'].median()]
        for _, row in overshipping_areas.iterrows():
            alerts.append({
                'area': row['area'],
                'alert_type': 'overshipping',
                'severity': 'low',
                'message': f"Significant overshipping of {row['overshipping']} units in {row['area']}. Consider reducing shipment quantities."
            })
        
        return alerts
