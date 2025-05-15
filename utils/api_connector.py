"""
API Connector module for integrating with external supply chain systems.
Provides connectors for SAP, Oracle, and other supply chain management systems.
"""

import requests
import pandas as pd
import streamlit as st
from datetime import datetime
import json
import time
import os
from requests.auth import HTTPBasicAuth

class APIConnector:
    """Base class for API connectors with common functionality."""
    
    def __init__(self, base_url, api_key=None, username=None, password=None):
        """
        Initialize the API connector.
        
        Parameters:
        -----------
        base_url : str
            Base URL for the API
        api_key : str, optional
            API key for authentication
        username : str, optional
            Username for basic authentication
        password : str, optional
            Password for basic authentication
        """
        self.base_url = base_url
        self.api_key = api_key
        self.username = username
        self.password = password
        self.session = requests.Session()
        
        # Set up authentication if provided
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        elif username and password:
            self.session.auth = HTTPBasicAuth(username, password)
    
    def make_request(self, endpoint, method="GET", params=None, data=None, headers=None):
        """
        Make an API request.
        
        Parameters:
        -----------
        endpoint : str
            API endpoint to call
        method : str
            HTTP method (GET, POST, PUT, DELETE)
        params : dict, optional
            Query parameters
        data : dict, optional
            Request body for POST/PUT requests
        headers : dict, optional
            Additional headers
            
        Returns:
        --------
        dict
            Response data
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        default_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if headers:
            default_headers.update(headers)
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params, headers=default_headers)
            elif method.upper() == "POST":
                response = self.session.post(url, params=params, json=data, headers=default_headers)
            elif method.upper() == "PUT":
                response = self.session.put(url, params=params, json=data, headers=default_headers)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, params=params, headers=default_headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            if response.content:
                return response.json()
            return {}
            
        except requests.exceptions.RequestException as e:
            st.error(f"API request error: {str(e)}")
            return {"error": str(e)}
    
    def to_dataframe(self, data, record_path=None):
        """
        Convert API response to pandas DataFrame.
        
        Parameters:
        -----------
        data : dict or list
            API response data
        record_path : str or list, optional
            Path to the records in the JSON structure
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the data
        """
        try:
            if record_path:
                if isinstance(record_path, str):
                    current = data
                    for key in record_path.split('.'):
                        current = current[key]
                    return pd.json_normalize(current)
                else:
                    return pd.json_normalize(data, record_path=record_path)
            elif isinstance(data, list):
                return pd.json_normalize(data)
            elif isinstance(data, dict):
                if "results" in data:
                    return pd.json_normalize(data["results"])
                elif "data" in data:
                    return pd.json_normalize(data["data"])
                elif "items" in data:
                    return pd.json_normalize(data["items"])
                else:
                    return pd.json_normalize([data])
            else:
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error converting response to DataFrame: {str(e)}")
            return pd.DataFrame()

class SAPConnector(APIConnector):
    """
    SAP S/4HANA OData API connector for retrieving supply chain data.
    """
    
    def __init__(self, base_url, api_key=None, username=None, password=None):
        """Initialize the SAP connector."""
        super().__init__(base_url, api_key, username, password)
        self.system_type = "SAP"
    
    def get_shipment_data(self, start_date=None, end_date=None, limit=1000):
        """
        Get shipment data from SAP.
        
        Parameters:
        -----------
        start_date : str, optional
            Start date in format 'YYYY-MM-DD'
        end_date : str, optional
            End date in format 'YYYY-MM-DD'
        limit : int, optional
            Maximum number of records to return
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing shipment data
        """
        endpoint = "/sap/opu/odata/sap/API_SHIPMENT_DOCUMENT_SRV/A_ShipmentDocument"
        params = {"$top": limit, "$format": "json"}
        
        if start_date:
            params["$filter"] = f"ShipmentDocumentDate ge datetime'{start_date}T00:00:00'"
            if end_date:
                params["$filter"] += f" and ShipmentDocumentDate le datetime'{end_date}T23:59:59'"
        
        response = self.make_request(endpoint, params=params)
        
        if "error" in response:
            return pd.DataFrame()
        
        # Transform SAP-specific fields to our standard format
        df = self.to_dataframe(response, "d.results")
        
        if not df.empty:
            # Map SAP fields to our standard fields
            field_mapping = {
                "ShipmentDocumentDate": "shipment_date",
                "DestinationLocation": "area",
                "MaterialID": "product_category",
                "ShipmentQuantity": "quantity_sent",
                "ShipmentCosts": "shipment_cost",
                "TransportationMode": "transportation_mode",
                "PlannedTransitTimeDuration": "delivery_time_days"
            }
            
            # Rename and select relevant columns
            if all(col in df.columns for col in field_mapping.keys()):
                df = df.rename(columns=field_mapping)
                return df[list(field_mapping.values())]
        
        return df
    
    def get_sales_data(self, start_date=None, end_date=None, limit=1000):
        """
        Get sales data from SAP.
        
        Parameters:
        -----------
        start_date : str, optional
            Start date in format 'YYYY-MM-DD'
        end_date : str, optional
            End date in format 'YYYY-MM-DD'
        limit : int, optional
            Maximum number of records to return
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing sales data
        """
        endpoint = "/sap/opu/odata/sap/API_SALES_ORDER_SRV/A_SalesOrder"
        params = {"$top": limit, "$format": "json"}
        
        if start_date:
            params["$filter"] = f"CreationDate ge datetime'{start_date}T00:00:00'"
            if end_date:
                params["$filter"] += f" and CreationDate le datetime'{end_date}T23:59:59'"
        
        response = self.make_request(endpoint, params=params)
        
        if "error" in response:
            return pd.DataFrame()
        
        df = self.to_dataframe(response, "d.results")
        
        if not df.empty:
            # Map SAP fields to our standard fields
            field_mapping = {
                "CreationDate": "sale_date",
                "SoldToParty": "area",
                "ProductCategory": "product_category",
                "TotalQuantity": "units_sold",
                "TotalNetAmount": "revenue",
                "DistributionChannel": "customer_segment",
                "GrossMargin": "profit_margin"
            }
            
            # Rename and select relevant columns
            if all(col in df.columns for col in field_mapping.keys()):
                df = df.rename(columns=field_mapping)
                return df[list(field_mapping.values())]
        
        return df

class OracleConnector(APIConnector):
    """
    Oracle SCM Cloud REST API connector for retrieving supply chain data.
    """
    
    def __init__(self, base_url, api_key=None, username=None, password=None):
        """Initialize the Oracle connector."""
        super().__init__(base_url, api_key, username, password)
        self.system_type = "Oracle"
    
    def get_shipment_data(self, start_date=None, end_date=None, limit=1000):
        """
        Get shipment data from Oracle SCM Cloud.
        
        Parameters:
        -----------
        start_date : str, optional
            Start date in format 'YYYY-MM-DD'
        end_date : str, optional
            End date in format 'YYYY-MM-DD'
        limit : int, optional
            Maximum number of records to return
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing shipment data
        """
        endpoint = "/fscmRestApi/resources/11.13.18.05/shipments"
        params = {"limit": limit, "fields": "ALL"}
        
        if start_date:
            params["q"] = f"ShipmentDate >= '{start_date}'"
            if end_date:
                params["q"] += f" AND ShipmentDate <= '{end_date}'"
        
        response = self.make_request(endpoint, params=params)
        
        if "error" in response:
            return pd.DataFrame()
        
        df = self.to_dataframe(response, "items")
        
        if not df.empty:
            # Map Oracle fields to our standard fields
            field_mapping = {
                "ShipmentDate": "shipment_date",
                "DestinationLocation": "area",
                "ItemCategory": "product_category",
                "ShipmentQuantity": "quantity_sent",
                "ShipmentCost": "shipment_cost",
                "ModeOfTransport": "transportation_mode",
                "EstimatedTransitDays": "delivery_time_days"
            }
            
            # Rename and select relevant columns
            if all(col in df.columns for col in field_mapping.keys()):
                df = df.rename(columns=field_mapping)
                return df[list(field_mapping.values())]
        
        return df
    
    def get_sales_data(self, start_date=None, end_date=None, limit=1000):
        """
        Get sales data from Oracle SCM Cloud.
        
        Parameters:
        -----------
        start_date : str, optional
            Start date in format 'YYYY-MM-DD'
        end_date : str, optional
            End date in format 'YYYY-MM-DD'
        limit : int, optional
            Maximum number of records to return
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing sales data
        """
        endpoint = "/fscmRestApi/resources/11.13.18.05/salesOrders"
        params = {"limit": limit, "fields": "ALL"}
        
        if start_date:
            params["q"] = f"OrderDate >= '{start_date}'"
            if end_date:
                params["q"] += f" AND OrderDate <= '{end_date}'"
        
        response = self.make_request(endpoint, params=params)
        
        if "error" in response:
            return pd.DataFrame()
        
        df = self.to_dataframe(response, "items")
        
        if not df.empty:
            # Map Oracle fields to our standard fields
            field_mapping = {
                "OrderDate": "sale_date",
                "ShipToLocation": "area",
                "ProductCategory": "product_category",
                "OrderedQuantity": "units_sold",
                "TotalAmount": "revenue",
                "CustomerSegment": "customer_segment",
                "ProfitMargin": "profit_margin"
            }
            
            # Rename and select relevant columns
            if all(col in df.columns for col in field_mapping.keys()):
                df = df.rename(columns=field_mapping)
                return df[list(field_mapping.values())]
        
        return df

class GenericRestConnector(APIConnector):
    """
    Generic REST API connector for retrieving supply chain data from any REST API.
    Configurable with custom endpoint mappings.
    """
    
    def __init__(self, base_url, api_key=None, username=None, password=None, 
                 endpoints=None, field_mappings=None):
        """
        Initialize the generic REST connector.
        
        Parameters:
        -----------
        base_url : str
            Base URL for the API
        api_key : str, optional
            API key for authentication
        username : str, optional
            Username for basic authentication
        password : str, optional
            Password for basic authentication
        endpoints : dict, optional
            Custom endpoint mappings for different data types
        field_mappings : dict, optional
            Custom field mappings for different data types
        """
        super().__init__(base_url, api_key, username, password)
        self.system_type = "Generic"
        
        # Default endpoints
        self.endpoints = {
            "shipment": "/shipments",
            "sales": "/sales",
            "inventory": "/inventory",
            "cost": "/costs"
        }
        
        # Default field mappings
        self.field_mappings = {
            "shipment": {
                "date": "shipment_date",
                "location": "area",
                "category": "product_category",
                "quantity": "quantity_sent",
                "cost": "shipment_cost",
                "transport_mode": "transportation_mode",
                "transit_time": "delivery_time_days"
            },
            "sales": {
                "date": "sale_date",
                "location": "area",
                "category": "product_category",
                "quantity": "units_sold",
                "revenue": "revenue",
                "segment": "customer_segment",
                "margin": "profit_margin"
            }
        }
        
        # Update with custom values if provided
        if endpoints:
            self.endpoints.update(endpoints)
        
        if field_mappings:
            for key, value in field_mappings.items():
                if key in self.field_mappings:
                    self.field_mappings[key].update(value)
                else:
                    self.field_mappings[key] = value
    
    def get_data(self, data_type, start_date=None, end_date=None, limit=1000, record_path=None):
        """
        Get data of specified type from the API.
        
        Parameters:
        -----------
        data_type : str
            Type of data to retrieve ('shipment', 'sales', 'inventory', 'cost')
        start_date : str, optional
            Start date in format 'YYYY-MM-DD'
        end_date : str, optional
            End date in format 'YYYY-MM-DD'
        limit : int, optional
            Maximum number of records to return
        record_path : str, optional
            Path to the records in the JSON structure
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the requested data
        """
        if data_type not in self.endpoints:
            st.error(f"Unknown data type: {data_type}")
            return pd.DataFrame()
        
        endpoint = self.endpoints[data_type]
        params = {"limit": limit}
        
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        response = self.make_request(endpoint, params=params)
        
        if "error" in response:
            return pd.DataFrame()
        
        df = self.to_dataframe(response, record_path)
        
        if not df.empty and data_type in self.field_mappings:
            # Create reverse mapping
            reverse_mapping = {v: k for k, v in self.field_mappings[data_type].items()}
            
            # Identify columns to rename
            cols_to_rename = {col: reverse_mapping[col] for col in reverse_mapping.keys() 
                             if col in df.columns}
            
            if cols_to_rename:
                df = df.rename(columns=cols_to_rename)
        
        return df
    
    def get_shipment_data(self, start_date=None, end_date=None, limit=1000):
        """
        Get shipment data from the API.
        
        Parameters:
        -----------
        start_date : str, optional
            Start date in format 'YYYY-MM-DD'
        end_date : str, optional
            End date in format 'YYYY-MM-DD'
        limit : int, optional
            Maximum number of records to return
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing shipment data
        """
        return self.get_data("shipment", start_date, end_date, limit)
    
    def get_sales_data(self, start_date=None, end_date=None, limit=1000):
        """
        Get sales data from the API.
        
        Parameters:
        -----------
        start_date : str, optional
            Start date in format 'YYYY-MM-DD'
        end_date : str, optional
            End date in format 'YYYY-MM-DD'
        limit : int, optional
            Maximum number of records to return
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing sales data
        """
        return self.get_data("sales", start_date, end_date, limit)

def get_connector_by_type(system_type, base_url, api_key=None, username=None, password=None):
    """
    Factory function to get the appropriate connector based on system type.
    
    Parameters:
    -----------
    system_type : str
        Type of system ('SAP', 'Oracle', 'Generic')
    base_url : str
        Base URL for the API
    api_key : str, optional
        API key for authentication
    username : str, optional
        Username for basic authentication
    password : str, optional
        Password for basic authentication
        
    Returns:
    --------
    APIConnector
        The appropriate connector instance
    """
    if system_type.upper() == "SAP":
        return SAPConnector(base_url, api_key, username, password)
    elif system_type.upper() == "ORACLE":
        return OracleConnector(base_url, api_key, username, password)
    else:
        return GenericRestConnector(base_url, api_key, username, password)

def test_connection(connector):
    """
    Test the connection to the API.
    
    Parameters:
    -----------
    connector : APIConnector
        The connector to test
        
    Returns:
    --------
    bool
        True if connection is successful, False otherwise
    """
    try:
        # Try a simple request to test the connection
        response = connector.make_request("")
        return "error" not in response
    except:
        return False