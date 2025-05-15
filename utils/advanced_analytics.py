"""
Advanced Analytics module for the Supply Chain Analytics tool.
Provides advanced algorithms for inventory optimization, 
transportation route optimization, risk assessment, and what-if analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import streamlit as st
import networkx as nx
from datetime import datetime, timedelta
import random
from itertools import permutations

class InventoryOptimizer:
    """
    Class for inventory optimization algorithms.
    Implements EOQ, safety stock, and multi-echelon inventory optimization models.
    """
    
    def __init__(self, inventory_data, sales_data, cost_data):
        """
        Initialize the inventory optimizer.
        
        Parameters:
        -----------
        inventory_data : pandas.DataFrame
            DataFrame containing inventory data
        sales_data : pandas.DataFrame
            DataFrame containing sales data
        cost_data : pandas.DataFrame
            DataFrame containing cost data
        """
        self.inventory_data = inventory_data
        self.sales_data = sales_data
        self.cost_data = cost_data
    
    def calculate_economic_order_quantity(self, annual_demand, ordering_cost, holding_cost_percentage, unit_cost):
        """
        Calculate the Economic Order Quantity (EOQ).
        
        Parameters:
        -----------
        annual_demand : float
            Annual demand in units
        ordering_cost : float
            Cost per order
        holding_cost_percentage : float
            Annual holding cost as a percentage of unit cost
        unit_cost : float
            Cost per unit
            
        Returns:
        --------
        float
            The optimal order quantity
        """
        holding_cost = unit_cost * holding_cost_percentage
        
        # EOQ formula: sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        
        return eoq
    
    def calculate_reorder_point(self, avg_daily_demand, lead_time_days, service_level=0.95):
        """
        Calculate the reorder point based on lead time and service level.
        
        Parameters:
        -----------
        avg_daily_demand : float
            Average daily demand in units
        lead_time_days : float
            Lead time in days
        service_level : float
            Desired service level (0-1)
            
        Returns:
        --------
        float
            The reorder point
        """
        # Calculate safety stock
        z_score = 1.645  # Z-score for 95% service level
        if service_level == 0.99:
            z_score = 2.326
        elif service_level == 0.90:
            z_score = 1.282
        
        # Assuming a standard deviation of 30% of average daily demand
        std_dev = avg_daily_demand * 0.3
        
        safety_stock = z_score * std_dev * np.sqrt(lead_time_days)
        
        # Reorder point = lead time demand + safety stock
        reorder_point = (avg_daily_demand * lead_time_days) + safety_stock
        
        return reorder_point
    
    def optimize_inventory_levels(self, area=None):
        """
        Calculate optimal inventory levels for products in specified area.
        
        Parameters:
        -----------
        area : str, optional
            Specific area to optimize, or all areas if None
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with optimal inventory levels
        """
        # Filter data by area if specified
        if area:
            inventory = self.inventory_data[self.inventory_data['area'] == area]
            sales = self.sales_data[self.sales_data['area'] == area]
            costs = self.cost_data[self.cost_data['area'] == area]
        else:
            inventory = self.inventory_data
            sales = self.sales_data
            costs = self.cost_data
        
        # Prepare the results DataFrame
        results = []
        
        # Group by area and product category
        for (area_name, product_cat), area_inventory in inventory.groupby(['area', 'product_category']):
            # Get sales data for this area and product
            area_sales = sales[(sales['area'] == area_name) & 
                               (sales['product_category'] == product_cat)]
            
            # Calculate annual demand
            if not area_sales.empty:
                annual_demand = area_sales['units_sold'].sum()
                avg_unit_revenue = area_sales['revenue'].sum() / area_sales['units_sold'].sum()
            else:
                annual_demand = 0
                avg_unit_revenue = 0
            
            # Get cost data for this area
            area_cost = costs[costs['area'] == area_name]
            
            # Assumptions and parameters
            ordering_cost = 100  # Fixed ordering cost
            holding_cost_percentage = 0.20  # Annual holding cost as percentage of unit cost
            lead_time_days = 14  # Lead time in days
            service_level = 0.95  # Service level (95%)
            
            # Calculate average daily demand
            avg_daily_demand = annual_demand / 365 if annual_demand > 0 else 0
            
            # Calculate the cost per unit (using a placeholder)
            if not area_inventory.empty:
                unit_cost = avg_unit_revenue * 0.7  # Assuming 30% profit margin
            else:
                unit_cost = 0
                
            # Skip calculation if no demand or costs
            if annual_demand == 0 or unit_cost == 0:
                continue
                
            # Calculate EOQ and reorder point
            eoq = self.calculate_economic_order_quantity(
                annual_demand, ordering_cost, holding_cost_percentage, unit_cost
            )
            
            reorder_point = self.calculate_reorder_point(
                avg_daily_demand, lead_time_days, service_level
            )
            
            # Calculate max inventory level
            max_inventory = reorder_point + eoq
            
            # Calculate safety stock
            safety_stock = reorder_point - (avg_daily_demand * lead_time_days)
            
            # Calculate average inventory
            avg_inventory = (eoq / 2) + safety_stock
            
            # Calculate annual holding cost
            annual_holding_cost = avg_inventory * unit_cost * holding_cost_percentage
            
            # Calculate annual ordering cost
            order_frequency = annual_demand / eoq if eoq > 0 else 0
            annual_ordering_cost = ordering_cost * order_frequency
            
            # Calculate total annual inventory cost
            total_annual_cost = annual_holding_cost + annual_ordering_cost
            
            # Calculate Inventory Turnover Ratio
            inventory_turnover = annual_demand / avg_inventory if avg_inventory > 0 else 0
            
            # Create result row
            result = {
                'area': area_name,
                'product_category': product_cat,
                'annual_demand': annual_demand,
                'optimal_order_quantity': round(eoq, 2),
                'reorder_point': round(reorder_point, 2),
                'safety_stock': round(safety_stock, 2),
                'max_inventory_level': round(max_inventory, 2),
                'avg_inventory_level': round(avg_inventory, 2),
                'order_frequency_per_year': round(order_frequency, 2),
                'inventory_turnover_ratio': round(inventory_turnover, 2),
                'annual_holding_cost': round(annual_holding_cost, 2),
                'annual_ordering_cost': round(annual_ordering_cost, 2),
                'total_annual_inventory_cost': round(total_annual_cost, 2)
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def create_inventory_optimization_visualization(self, optimization_results):
        """
        Create visualizations for inventory optimization results.
        
        Parameters:
        -----------
        optimization_results : pandas.DataFrame
            DataFrame with optimization results
            
        Returns:
        --------
        dict
            Dictionary with Plotly figures
        """
        figures = {}
        
        if optimization_results.empty:
            return figures
        
        # Inventory costs by area
        fig_costs = px.bar(
            optimization_results,
            x='area',
            y=['annual_holding_cost', 'annual_ordering_cost'],
            title='Annual Inventory Costs by Area',
            barmode='stack',
            labels={'value': 'Cost', 'variable': 'Cost Type'}
        )
        figures['costs'] = fig_costs
        
        # Inventory turnover ratio
        fig_turnover = px.bar(
            optimization_results,
            x='area',
            y='inventory_turnover_ratio',
            color='product_category',
            title='Inventory Turnover Ratio by Area and Product',
            labels={'inventory_turnover_ratio': 'Turnover Ratio', 'area': 'Area'}
        )
        figures['turnover'] = fig_turnover
        
        # Optimal order quantities
        fig_eoq = px.scatter(
            optimization_results,
            x='annual_demand',
            y='optimal_order_quantity',
            color='area',
            size='total_annual_inventory_cost',
            hover_name='product_category',
            title='Optimal Order Quantity vs Annual Demand',
            labels={
                'annual_demand': 'Annual Demand',
                'optimal_order_quantity': 'Optimal Order Quantity (EOQ)'
            }
        )
        figures['eoq'] = fig_eoq
        
        # Safety stock vs Reorder point
        fig_safety = px.scatter(
            optimization_results,
            x='safety_stock',
            y='reorder_point',
            color='area',
            size='annual_demand',
            hover_name='product_category',
            title='Safety Stock vs Reorder Point',
            labels={
                'safety_stock': 'Safety Stock',
                'reorder_point': 'Reorder Point'
            }
        )
        figures['safety'] = fig_safety
        
        return figures

class RouteOptimizer:
    """
    Class for transportation route optimization.
    Implements algorithms for solving vehicle routing problems.
    """
    
    def __init__(self, shipment_data, area_coordinates=None):
        """
        Initialize the route optimizer.
        
        Parameters:
        -----------
        shipment_data : pandas.DataFrame
            DataFrame containing shipment data
        area_coordinates : pandas.DataFrame, optional
            DataFrame with area names and their coordinates (latitude, longitude)
        """
        self.shipment_data = shipment_data
        self.area_coordinates = area_coordinates
        
        # If no coordinates provided, create some placeholder coordinates
        if self.area_coordinates is None:
            self._create_placeholder_coordinates()
    
    def _create_placeholder_coordinates(self):
        """Create placeholder coordinates for areas in the data."""
        areas = self.shipment_data['area'].unique()
        
        coordinates = []
        for i, area in enumerate(areas):
            # Generate some dummy coordinates for demonstration
            lat = 40 + np.sin(i * 0.7) * 10
            lon = -90 + np.cos(i * 0.7) * 20
            
            coordinates.append({
                'area': area,
                'latitude': lat,
                'longitude': lon
            })
        
        self.area_coordinates = pd.DataFrame(coordinates)
    
    def calculate_distance_matrix(self):
        """
        Calculate the distance matrix between areas.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing distances between areas
        """
        # Extract coordinates
        areas = self.area_coordinates['area'].tolist()
        coords = self.area_coordinates[['latitude', 'longitude']].values
        
        # Calculate Euclidean distances
        distances = squareform(pdist(coords, metric='euclidean'))
        
        # Create distance matrix DataFrame
        distance_matrix = pd.DataFrame(distances, index=areas, columns=areas)
        
        return distance_matrix
    
    def optimize_route(self, origin, destinations=None, algorithm='tsp'):
        """
        Optimize a delivery route.
        
        Parameters:
        -----------
        origin : str
            Origin area name
        destinations : list, optional
            List of destination area names, or all areas if None
        algorithm : str
            Algorithm to use ('tsp', 'vrp', 'nearest_neighbor')
            
        Returns:
        --------
        dict
            Dictionary with optimized route information
        """
        # Get distance matrix
        distance_matrix = self.calculate_distance_matrix()
        
        # If no destinations specified, use all areas except origin
        if destinations is None:
            destinations = [area for area in distance_matrix.columns if area != origin]
        
        # Choose the appropriate algorithm
        if algorithm == 'tsp' or len(destinations) < 10:
            return self._solve_tsp(origin, destinations, distance_matrix)
        elif algorithm == 'vrp':
            return self._solve_vrp(origin, destinations, distance_matrix)
        else:  # nearest_neighbor
            return self._solve_nearest_neighbor(origin, destinations, distance_matrix)
    
    def _solve_tsp(self, origin, destinations, distance_matrix):
        """
        Solve the Traveling Salesman Problem to find the optimal route.
        For small problems, uses brute force to find the exact solution.
        
        Parameters:
        -----------
        origin : str
            Origin area name
        destinations : list
            List of destination area names
        distance_matrix : pandas.DataFrame
            Distance matrix between areas
            
        Returns:
        --------
        dict
            Dictionary with optimized route information
        """
        # Create a list of all points (origin + destinations)
        all_points = [origin] + destinations
        
        # If the problem is small enough, use brute force exact algorithm
        if len(destinations) <= 8:
            # Generate all possible permutations of destinations
            all_permutations = list(permutations(destinations))
            
            min_distance = float('inf')
            best_route = None
            
            # Evaluate each permutation
            for perm in all_permutations:
                # Complete route: origin -> permutation -> origin
                route = [origin] + list(perm) + [origin]
                
                # Calculate total distance
                total_distance = 0
                for i in range(len(route) - 1):
                    from_area = route[i]
                    to_area = route[i + 1]
                    total_distance += distance_matrix.loc[from_area, to_area]
                
                # Update best route if better
                if total_distance < min_distance:
                    min_distance = total_distance
                    best_route = route
            
            result = {
                'route': best_route,
                'distance': min_distance,
                'algorithm': 'exact_tsp'
            }
        else:
            # For larger problems, use nearest neighbor heuristic
            result = self._solve_nearest_neighbor(origin, destinations, distance_matrix)
            result['algorithm'] = 'nearest_neighbor_tsp'
        
        return result
    
    def _solve_nearest_neighbor(self, origin, destinations, distance_matrix):
        """
        Solve the routing problem using the nearest neighbor heuristic.
        
        Parameters:
        -----------
        origin : str
            Origin area name
        destinations : list
            List of destination area names
        distance_matrix : pandas.DataFrame
            Distance matrix between areas
            
        Returns:
        --------
        dict
            Dictionary with optimized route information
        """
        # Initialize route
        route = [origin]
        unvisited = destinations.copy()
        total_distance = 0
        
        # Current position
        current = origin
        
        # Visit all destinations
        while unvisited:
            # Find the nearest unvisited destination
            min_distance = float('inf')
            nearest = None
            
            for dest in unvisited:
                dist = distance_matrix.loc[current, dest]
                if dist < min_distance:
                    min_distance = dist
                    nearest = dest
            
            # Update route
            route.append(nearest)
            unvisited.remove(nearest)
            total_distance += min_distance
            current = nearest
        
        # Return to origin
        route.append(origin)
        total_distance += distance_matrix.loc[current, origin]
        
        return {
            'route': route,
            'distance': total_distance,
            'algorithm': 'nearest_neighbor'
        }
    
    def _solve_vrp(self, origin, destinations, distance_matrix):
        """
        Solve a Vehicle Routing Problem.
        This is a simplified implementation using clustering.
        
        Parameters:
        -----------
        origin : str
            Origin area name
        destinations : list
            List of destination area names
        distance_matrix : pandas.DataFrame
            Distance matrix between areas
            
        Returns:
        --------
        dict
            Dictionary with optimized route information
        """
        # Number of vehicles/routes
        num_vehicles = min(5, len(destinations) // 3 + 1)
        
        # Get coordinates for all destinations
        all_coords = []
        for dest in destinations:
            area_row = self.area_coordinates[self.area_coordinates['area'] == dest]
            if not area_row.empty:
                coords = area_row[['latitude', 'longitude']].values[0]
                all_coords.append(coords)
        
        # If we don't have enough coordinates, fallback to nearest neighbor
        if len(all_coords) < len(destinations):
            return self._solve_nearest_neighbor(origin, destinations, distance_matrix)
        
        # Cluster destinations into groups for each vehicle
        if len(destinations) >= num_vehicles:
            kmeans = KMeans(n_clusters=num_vehicles, random_state=42)
            clusters = kmeans.fit_predict(all_coords)
        else:
            # If fewer destinations than vehicles, assign one destination per vehicle
            clusters = list(range(len(destinations)))
        
        # Group destinations by cluster
        cluster_destinations = {}
        for i, dest in enumerate(destinations):
            cluster = clusters[i] if i < len(clusters) else 0
            if cluster not in cluster_destinations:
                cluster_destinations[cluster] = []
            cluster_destinations[cluster].append(dest)
        
        # Solve TSP for each cluster
        routes = []
        total_distance = 0
        
        for cluster, dests in cluster_destinations.items():
            if dests:
                result = self._solve_tsp(origin, dests, distance_matrix)
                routes.append(result['route'])
                total_distance += result['distance']
        
        return {
            'routes': routes,
            'distance': total_distance,
            'algorithm': 'vrp_clustering'
        }
    
    def create_route_visualization(self, route_result):
        """
        Create a visualization of the optimized route.
        
        Parameters:
        -----------
        route_result : dict
            Result from optimize_route method
            
        Returns:
        --------
        plotly.graph_objects.Figure
            A Plotly figure with the route visualization
        """
        # Check if multiple routes (VRP) or single route (TSP)
        is_vrp = 'routes' in route_result
        
        # Create a base map
        fig = go.Figure()
        
        # Add nodes (areas)
        for _, row in self.area_coordinates.iterrows():
            fig.add_trace(go.Scattergeo(
                lon=[row['longitude']],
                lat=[row['latitude']],
                text=[row['area']],
                mode='markers',
                marker=dict(
                    size=10,
                    color='blue',
                    line=dict(width=1, color='black')
                ),
                name=row['area']
            ))
        
        # Highlight the origin
        if not is_vrp and route_result['route']:
            origin = route_result['route'][0]
            origin_coords = self.area_coordinates[self.area_coordinates['area'] == origin]
            if not origin_coords.empty:
                fig.add_trace(go.Scattergeo(
                    lon=[origin_coords['longitude'].values[0]],
                    lat=[origin_coords['latitude'].values[0]],
                    text=[origin],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='red',
                        line=dict(width=2, color='black')
                    ),
                    name='Origin'
                ))
        
        # Add the routes
        route_colors = px.colors.qualitative.Plotly
        
        if is_vrp:
            # Add multiple routes for VRP
            for i, route in enumerate(route_result['routes']):
                route_coords = []
                for area in route:
                    area_row = self.area_coordinates[self.area_coordinates['area'] == area]
                    if not area_row.empty:
                        lat = area_row['latitude'].values[0]
                        lon = area_row['longitude'].values[0]
                        route_coords.append((lat, lon))
                
                # Create lines between points
                if route_coords:
                    lats, lons = zip(*route_coords)
                    color = route_colors[i % len(route_colors)]
                    
                    fig.add_trace(go.Scattergeo(
                        lon=lons,
                        lat=lats,
                        mode='lines',
                        line=dict(width=2, color=color),
                        name=f'Route {i+1}'
                    ))
        else:
            # Add single route for TSP
            route_coords = []
            for area in route_result['route']:
                area_row = self.area_coordinates[self.area_coordinates['area'] == area]
                if not area_row.empty:
                    lat = area_row['latitude'].values[0]
                    lon = area_row['longitude'].values[0]
                    route_coords.append((lat, lon))
            
            # Create lines between points
            if route_coords:
                lats, lons = zip(*route_coords)
                
                fig.add_trace(go.Scattergeo(
                    lon=lons,
                    lat=lats,
                    mode='lines',
                    line=dict(width=2, color='red'),
                    name='Optimized Route'
                ))
        
        # Update layout
        fig.update_layout(
            title=f"Optimized Delivery Route ({route_result['algorithm']})",
            geo=dict(
                projection_type='natural earth',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)'
            ),
            height=600
        )
        
        return fig

class RiskAssessor:
    """
    Class for supply chain risk assessment.
    Evaluates various risk factors and provides risk scores.
    """
    
    def __init__(self, shipment_data, sales_data, inventory_data, cost_data):
        """
        Initialize the risk assessor.
        
        Parameters:
        -----------
        shipment_data : pandas.DataFrame
            DataFrame containing shipment data
        sales_data : pandas.DataFrame
            DataFrame containing sales data
        inventory_data : pandas.DataFrame
            DataFrame containing inventory data
        cost_data : pandas.DataFrame
            DataFrame containing cost data
        """
        self.shipment_data = shipment_data
        self.sales_data = sales_data
        self.inventory_data = inventory_data
        self.cost_data = cost_data
    
    def calculate_risk_scores(self):
        """
        Calculate risk scores for each area.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with risk assessment results
        """
        # Get unique areas
        areas = self.shipment_data['area'].unique()
        
        risk_results = []
        
        for area in areas:
            # Filter data for this area
            area_shipments = self.shipment_data[self.shipment_data['area'] == area]
            area_sales = self.sales_data[self.sales_data['area'] == area]
            area_inventory = self.inventory_data[self.inventory_data['area'] == area]
            area_costs = self.cost_data[self.cost_data['area'] == area]
            
            # Skip if no data
            if area_shipments.empty or area_sales.empty or area_inventory.empty:
                continue
            
            # Calculate delivery reliability
            if 'delivery_time_days' in area_shipments.columns:
                avg_delivery_time = area_shipments['delivery_time_days'].mean()
                delivery_time_std = area_shipments['delivery_time_days'].std()
                delivery_reliability_risk = min(delivery_time_std / avg_delivery_time if avg_delivery_time > 0 else 1, 1)
            else:
                delivery_reliability_risk = 0.5  # Default if no data
            
            # Calculate stockout risk
            if 'stockout_count' in area_inventory.columns and 'remaining_stock' in area_inventory.columns:
                stockout_frequency = area_inventory['stockout_count'].sum() / len(area_inventory)
                avg_stock = area_inventory['remaining_stock'].mean()
                
                # Calculate average daily demand
                if 'units_sold' in area_sales.columns:
                    total_sold = area_sales['units_sold'].sum()
                    avg_daily_demand = total_sold / 365  # Assuming a year of data
                else:
                    avg_daily_demand = 10  # Default
                
                days_of_supply = avg_stock / avg_daily_demand if avg_daily_demand > 0 else 30
                stockout_risk = min(stockout_frequency + (1 / days_of_supply), 1)
            else:
                stockout_risk = 0.5  # Default if no data
            
            # Calculate demand volatility
            if 'units_sold' in area_sales.columns:
                sales_cv = area_sales['units_sold'].std() / area_sales['units_sold'].mean() if area_sales['units_sold'].mean() > 0 else 1
                demand_volatility_risk = min(sales_cv, 1)
            else:
                demand_volatility_risk = 0.5  # Default if no data
            
            # Calculate cost risk
            if not area_costs.empty and 'transportation_cost' in area_costs.columns and 'warehouse_cost' in area_costs.columns:
                total_costs = area_costs['transportation_cost'].sum() + area_costs['warehouse_cost'].sum()
                
                # Calculate total revenue
                if 'revenue' in area_sales.columns:
                    total_revenue = area_sales['revenue'].sum()
                    cost_to_revenue = total_costs / total_revenue if total_revenue > 0 else 1
                    cost_risk = min(cost_to_revenue, 1)
                else:
                    cost_risk = 0.5  # Default if no data
            else:
                cost_risk = 0.5  # Default if no data
            
            # Calculate supplier concentration risk
            # (Using a dummy calculation since we don't have supplier data)
            supplier_concentration_risk = 0.5
            
            # Calculate transportation risk
            # (Using a dummy calculation based on delivery times)
            if 'delivery_time_days' in area_shipments.columns:
                long_deliveries = (area_shipments['delivery_time_days'] > 10).mean()
                transportation_risk = min(long_deliveries + 0.2, 1)
            else:
                transportation_risk = 0.5  # Default if no data
            
            # Calculate overall risk score (weighted average)
            weights = {
                'delivery_reliability_risk': 0.15,
                'stockout_risk': 0.20,
                'demand_volatility_risk': 0.20,
                'cost_risk': 0.15,
                'supplier_concentration_risk': 0.15,
                'transportation_risk': 0.15
            }
            
            overall_risk = (
                weights['delivery_reliability_risk'] * delivery_reliability_risk +
                weights['stockout_risk'] * stockout_risk +
                weights['demand_volatility_risk'] * demand_volatility_risk +
                weights['cost_risk'] * cost_risk +
                weights['supplier_concentration_risk'] * supplier_concentration_risk +
                weights['transportation_risk'] * transportation_risk
            )
            
            # Create result row
            result = {
                'area': area,
                'delivery_reliability_risk': round(delivery_reliability_risk, 2),
                'stockout_risk': round(stockout_risk, 2),
                'demand_volatility_risk': round(demand_volatility_risk, 2),
                'cost_risk': round(cost_risk, 2),
                'supplier_concentration_risk': round(supplier_concentration_risk, 2),
                'transportation_risk': round(transportation_risk, 2),
                'overall_risk_score': round(overall_risk, 2)
            }
            
            # Determine risk category
            if overall_risk < 0.3:
                result['risk_category'] = 'Low'
            elif overall_risk < 0.6:
                result['risk_category'] = 'Medium'
            else:
                result['risk_category'] = 'High'
            
            risk_results.append(result)
        
        return pd.DataFrame(risk_results)
    
    def create_risk_visualization(self, risk_results):
        """
        Create visualizations for risk assessment results.
        
        Parameters:
        -----------
        risk_results : pandas.DataFrame
            DataFrame with risk assessment results
            
        Returns:
        --------
        dict
            Dictionary with Plotly figures
        """
        figures = {}
        
        if risk_results.empty:
            return figures
        
        # Overall risk by area
        fig_overall = px.bar(
            risk_results,
            x='area',
            y='overall_risk_score',
            color='risk_category',
            title='Overall Risk Score by Area',
            color_discrete_map={
                'Low': 'green',
                'Medium': 'orange',
                'High': 'red'
            },
            labels={'overall_risk_score': 'Risk Score', 'area': 'Area'}
        )
        figures['overall'] = fig_overall
        
        # Risk radar chart
        risk_categories = [
            'delivery_reliability_risk', 'stockout_risk', 'demand_volatility_risk',
            'cost_risk', 'supplier_concentration_risk', 'transportation_risk'
        ]
        
        # Prepare data for radar chart
        fig_radar = go.Figure()
        
        for _, row in risk_results.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[cat] for cat in risk_categories],
                theta=[cat.replace('_risk', '').replace('_', ' ').title() for cat in risk_categories],
                fill='toself',
                name=row['area']
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title='Risk Assessment Radar Chart',
            showlegend=True
        )
        figures['radar'] = fig_radar
        
        # Risk heatmap
        risk_data = risk_results[risk_categories + ['area']].set_index('area')
        
        # Rename columns for better display
        risk_data.columns = [col.replace('_risk', '').replace('_', ' ').title() for col in risk_data.columns]
        
        fig_heatmap = px.imshow(
            risk_data.T,
            text_auto=True,
            aspect='auto',
            color_continuous_scale='Reds',
            title='Risk Factors Heatmap by Area',
            labels=dict(x='Area', y='Risk Factor', color='Risk Score')
        )
        figures['heatmap'] = fig_heatmap
        
        return figures

class WhatIfAnalyzer:
    """
    Class for what-if analysis of supply chain scenarios.
    Allows simulation of different scenarios and their impacts.
    """
    
    def __init__(self, shipment_data, sales_data, inventory_data, cost_data):
        """
        Initialize the what-if analyzer.
        
        Parameters:
        -----------
        shipment_data : pandas.DataFrame
            DataFrame containing shipment data
        sales_data : pandas.DataFrame
            DataFrame containing sales data
        inventory_data : pandas.DataFrame
            DataFrame containing inventory data
        cost_data : pandas.DataFrame
            DataFrame containing cost data
        """
        self.original_shipment_data = shipment_data.copy()
        self.original_sales_data = sales_data.copy()
        self.original_inventory_data = inventory_data.copy()
        self.original_cost_data = cost_data.copy()
        
        # Current working copies
        self.shipment_data = shipment_data.copy()
        self.sales_data = sales_data.copy()
        self.inventory_data = inventory_data.copy()
        self.cost_data = cost_data.copy()
    
    def reset_scenario(self):
        """Reset to original data."""
        self.shipment_data = self.original_shipment_data.copy()
        self.sales_data = self.original_sales_data.copy()
        self.inventory_data = self.original_inventory_data.copy()
        self.cost_data = self.original_cost_data.copy()
    
    def adjust_demand(self, area=None, product_category=None, change_percent=0):
        """
        Adjust demand by a percentage.
        
        Parameters:
        -----------
        area : str, optional
            Area to adjust, or all areas if None
        product_category : str, optional
            Product category to adjust, or all categories if None
        change_percent : float
            Percentage change (-100 to 100)
            
        Returns:
        --------
        pandas.DataFrame
            Updated sales data
        """
        # Filter rows to adjust
        mask = pd.Series(True, index=self.sales_data.index)
        
        if area:
            mask &= (self.sales_data['area'] == area)
        
        if product_category:
            mask &= (self.sales_data['product_category'] == product_category)
        
        # Calculate multiplier
        multiplier = 1 + (change_percent / 100)
        
        # Apply adjustment to units_sold and revenue
        if 'units_sold' in self.sales_data.columns:
            self.sales_data.loc[mask, 'units_sold'] = self.sales_data.loc[mask, 'units_sold'] * multiplier
        
        if 'revenue' in self.sales_data.columns:
            self.sales_data.loc[mask, 'revenue'] = self.sales_data.loc[mask, 'revenue'] * multiplier
        
        return self.sales_data
    
    def adjust_costs(self, cost_type, area=None, change_percent=0):
        """
        Adjust costs by a percentage.
        
        Parameters:
        -----------
        cost_type : str
            Type of cost to adjust ('transportation', 'warehouse', 'handling', 'shipment')
        area : str, optional
            Area to adjust, or all areas if None
        change_percent : float
            Percentage change (-100 to 100)
            
        Returns:
        --------
        pandas.DataFrame
            Updated cost data or shipment data
        """
        # Calculate multiplier
        multiplier = 1 + (change_percent / 100)
        
        if cost_type in ['transportation', 'warehouse', 'handling']:
            # Filter rows to adjust in cost_data
            mask = pd.Series(True, index=self.cost_data.index)
            
            if area:
                mask &= (self.cost_data['area'] == area)
            
            # Apply adjustment
            cost_column = f"{cost_type}_cost"
            if cost_column in self.cost_data.columns:
                self.cost_data.loc[mask, cost_column] = self.cost_data.loc[mask, cost_column] * multiplier
            
            return self.cost_data
            
        elif cost_type == 'shipment':
            # Filter rows to adjust in shipment_data
            mask = pd.Series(True, index=self.shipment_data.index)
            
            if area:
                mask &= (self.shipment_data['area'] == area)
            
            # Apply adjustment
            if 'shipment_cost' in self.shipment_data.columns:
                self.shipment_data.loc[mask, 'shipment_cost'] = self.shipment_data.loc[mask, 'shipment_cost'] * multiplier
            
            return self.shipment_data
        
        else:
            return None
    
    def adjust_lead_time(self, area=None, change_days=0):
        """
        Adjust delivery lead time.
        
        Parameters:
        -----------
        area : str, optional
            Area to adjust, or all areas if None
        change_days : float
            Number of days to add or subtract
            
        Returns:
        --------
        pandas.DataFrame
            Updated shipment data
        """
        # Filter rows to adjust
        mask = pd.Series(True, index=self.shipment_data.index)
        
        if area:
            mask &= (self.shipment_data['area'] == area)
        
        # Apply adjustment
        if 'delivery_time_days' in self.shipment_data.columns:
            self.shipment_data.loc[mask, 'delivery_time_days'] = self.shipment_data.loc[mask, 'delivery_time_days'] + change_days
            # Ensure no negative values
            self.shipment_data.loc[mask, 'delivery_time_days'] = self.shipment_data.loc[mask, 'delivery_time_days'].apply(lambda x: max(0, x))
        
        return self.shipment_data
    
    def adjust_inventory(self, area=None, product_category=None, change_percent=0):
        """
        Adjust inventory levels.
        
        Parameters:
        -----------
        area : str, optional
            Area to adjust, or all areas if None
        product_category : str, optional
            Product category to adjust, or all categories if None
        change_percent : float
            Percentage change (-100 to 100)
            
        Returns:
        --------
        pandas.DataFrame
            Updated inventory data
        """
        # Filter rows to adjust
        mask = pd.Series(True, index=self.inventory_data.index)
        
        if area:
            mask &= (self.inventory_data['area'] == area)
        
        if product_category:
            mask &= (self.inventory_data['product_category'] == product_category)
        
        # Calculate multiplier
        multiplier = 1 + (change_percent / 100)
        
        # Apply adjustment
        if 'remaining_stock' in self.inventory_data.columns:
            self.inventory_data.loc[mask, 'remaining_stock'] = self.inventory_data.loc[mask, 'remaining_stock'] * multiplier
        
        if 'holding_cost' in self.inventory_data.columns:
            self.inventory_data.loc[mask, 'holding_cost'] = self.inventory_data.loc[mask, 'holding_cost'] * multiplier
        
        return self.inventory_data
    
    def calculate_scenario_metrics(self):
        """
        Calculate key metrics for the current scenario.
        
        Returns:
        --------
        dict
            Dictionary with calculated metrics
        """
        # Total sales metrics
        total_units_sold = self.sales_data['units_sold'].sum() if 'units_sold' in self.sales_data.columns else 0
        total_revenue = self.sales_data['revenue'].sum() if 'revenue' in self.sales_data.columns else 0
        
        # Total cost metrics
        total_shipment_cost = self.shipment_data['shipment_cost'].sum() if 'shipment_cost' in self.shipment_data.columns else 0
        
        total_transportation_cost = self.cost_data['transportation_cost'].sum() if 'transportation_cost' in self.cost_data.columns else 0
        total_warehouse_cost = self.cost_data['warehouse_cost'].sum() if 'warehouse_cost' in self.cost_data.columns else 0
        total_handling_cost = self.cost_data['handling_cost'].sum() if 'handling_cost' in self.cost_data.columns else 0
        
        total_costs = total_shipment_cost + total_transportation_cost + total_warehouse_cost + total_handling_cost
        
        # Inventory metrics
        total_inventory = self.inventory_data['remaining_stock'].sum() if 'remaining_stock' in self.inventory_data.columns else 0
        total_holding_cost = self.inventory_data['holding_cost'].sum() if 'holding_cost' in self.inventory_data.columns else 0
        
        # Calculate profit
        profit = total_revenue - total_costs - total_holding_cost
        
        # Calculate profit margin
        profit_margin = profit / total_revenue if total_revenue > 0 else 0
        
        # Calculate average delivery time
        avg_delivery_time = self.shipment_data['delivery_time_days'].mean() if 'delivery_time_days' in self.shipment_data.columns else 0
        
        # Calculate inventory turnover
        inventory_turnover = total_units_sold / total_inventory if total_inventory > 0 else 0
        
        return {
            'total_units_sold': total_units_sold,
            'total_revenue': total_revenue,
            'total_shipment_cost': total_shipment_cost,
            'total_transportation_cost': total_transportation_cost,
            'total_warehouse_cost': total_warehouse_cost,
            'total_handling_cost': total_handling_cost,
            'total_costs': total_costs,
            'total_inventory': total_inventory,
            'total_holding_cost': total_holding_cost,
            'profit': profit,
            'profit_margin': profit_margin,
            'avg_delivery_time': avg_delivery_time,
            'inventory_turnover': inventory_turnover
        }
    
    def compare_scenarios(self, scenarios):
        """
        Compare multiple scenarios.
        
        Parameters:
        -----------
        scenarios : dict
            Dictionary with scenario names as keys and metric dictionaries as values
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with scenario comparison
        """
        # Convert scenarios to DataFrame
        comparison = pd.DataFrame(scenarios).T
        
        # Format the numbers
        format_dict = {
            'total_units_sold': '{:,.0f}',
            'total_revenue': '${:,.2f}',
            'total_shipment_cost': '${:,.2f}',
            'total_transportation_cost': '${:,.2f}',
            'total_warehouse_cost': '${:,.2f}',
            'total_handling_cost': '${:,.2f}',
            'total_costs': '${:,.2f}',
            'total_inventory': '{:,.0f}',
            'total_holding_cost': '${:,.2f}',
            'profit': '${:,.2f}',
            'profit_margin': '{:.2%}',
            'avg_delivery_time': '{:.1f} days',
            'inventory_turnover': '{:.2f}x'
        }
        
        for col, fmt in format_dict.items():
            if col in comparison.columns:
                comparison[col] = comparison[col].apply(lambda x: fmt.format(x))
        
        return comparison
    
    def create_scenario_visualization(self, scenarios, metric='profit'):
        """
        Create visualization comparing scenarios.
        
        Parameters:
        -----------
        scenarios : dict
            Dictionary with scenario names as keys and metric dictionaries as values
        metric : str
            Metric to visualize
            
        Returns:
        --------
        plotly.graph_objects.Figure
            A Plotly figure with the scenario comparison
        """
        # Extract the metric from each scenario
        scenario_names = list(scenarios.keys())
        metric_values = [scenarios[name][metric] for name in scenario_names]
        
        # Choose color based on metric trend (higher is better or worse)
        if metric in ['profit', 'profit_margin', 'total_revenue', 'inventory_turnover']:
            colors = ['green' if val >= scenarios['Baseline'][metric] else 'red' for val in metric_values]
        else:
            colors = ['red' if val >= scenarios['Baseline'][metric] else 'green' for val in metric_values]
        
        # Create bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=scenario_names,
            y=metric_values,
            marker_color=colors
        ))
        
        # Format y-axis label based on metric
        if metric in ['profit', 'total_revenue', 'total_shipment_cost', 'total_costs', 'total_holding_cost']:
            y_label = f"{metric.replace('_', ' ').title()} ($)"
        elif metric == 'profit_margin':
            y_label = "Profit Margin (%)"
            # Convert to percentage for display
            fig.update_yaxes(tickformat='.0%')
        elif metric == 'inventory_turnover':
            y_label = "Inventory Turnover (times)"
        elif metric == 'avg_delivery_time':
            y_label = "Average Delivery Time (days)"
        else:
            y_label = metric.replace('_', ' ').title()
        
        # Add reference line for baseline
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=scenarios['Baseline'][metric],
            x1=len(scenario_names) - 0.5,
            y1=scenarios['Baseline'][metric],
            line=dict(color="black", width=2, dash="dash")
        )
        
        # Add annotations showing percentage change from baseline
        baseline_value = scenarios['Baseline'][metric]
        for i, val in enumerate(metric_values):
            if scenario_names[i] != 'Baseline' and baseline_value != 0:
                percent_change = (val - baseline_value) / baseline_value * 100
                sign = "+" if percent_change > 0 else ""
                
                fig.add_annotation(
                    x=i,
                    y=val,
                    text=f"{sign}{percent_change:.1f}%",
                    showarrow=True,
                    arrowhead=4,
                    ax=0,
                    ay=-40
                )
        
        # Update layout
        fig.update_layout(
            title=f"Comparison of {metric.replace('_', ' ').title()} Across Scenarios",
            xaxis_title="Scenario",
            yaxis_title=y_label,
            height=500
        )
        
        return fig