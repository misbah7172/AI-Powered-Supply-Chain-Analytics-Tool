"""
Database utilities for connecting to MySQL and performing operations.
This module enables the application to use a MySQL database (e.g., through XAMPP)
instead of CSV files for data storage and retrieval.
"""

import pandas as pd
import pymysql
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import streamlit as st

class DatabaseManager:
    """
    Class for managing database connections and operations.
    Provides utilities to connect to MySQL, execute queries, and convert results to dataframes.
    """
    
    def __init__(self, host="localhost", user="root", password="", database="supply_chain_analytics", port=3306):
        """
        Initialize database connection parameters.
        
        Parameters:
        -----------
        host : str
            Database server hostname (default: localhost for XAMPP)
        user : str
            Database username (default: root for XAMPP)
        password : str
            Database password (default: empty for XAMPP)
        database : str
            Database name to connect to
        port : int
            Database port (default: 3306 for MySQL)
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.connection = None
        self.engine = None
    
    def connect(self):
        """
        Establish a connection to the MySQL database.
        
        Returns:
        --------
        bool
            True if connection is successful, False otherwise.
        """
        connection_string = f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        
        try:
            self.engine = create_engine(connection_string)
            self.connection = self.engine.connect()
            return True
        except SQLAlchemyError as e:
            st.error(f"Database connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """Close the database connection if it exists."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def execute_query(self, query, params=None):
        """
        Execute a SQL query with optional parameters.
        
        Parameters:
        -----------
        query : str
            SQL query to execute
        params : dict, optional
            Parameters to substitute in the query
            
        Returns:
        --------
        result
            Query execution result
        """
        if not self.connection and not self.connect():
            return None
            
        try:
            if params:
                result = self.connection.execute(text(query), params)
            else:
                result = self.connection.execute(text(query))
            return result
        except SQLAlchemyError as e:
            st.error(f"Query execution error: {str(e)}")
            return None
        except AttributeError:
            st.error("Database connection is not properly established")
            if self.connect():
                # Try once more after reconnecting
                try:
                    if params:
                        result = self.connection.execute(text(query), params)
                    else:
                        result = self.connection.execute(text(query))
                    return result
                except Exception as e:
                    st.error(f"Query execution error after reconnect: {str(e)}")
            return None
    
    def query_to_dataframe(self, query, params=None):
        """
        Execute a SQL query and return the results as a pandas DataFrame.
        
        Parameters:
        -----------
        query : str
            SQL query to execute
        params : dict, optional
            Parameters to substitute in the query
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing query results
        """
        if not self.connection and not self.connect():
            return pd.DataFrame()
            
        try:
            if params:
                return pd.read_sql(text(query), self.engine, params=params)
            else:
                return pd.read_sql(text(query), self.engine)
        except SQLAlchemyError as e:
            st.error(f"DataFrame conversion error: {str(e)}")
            return pd.DataFrame()
    
    def create_tables(self):
        """
        Create the necessary tables in the database if they don't exist.
        """
        create_shipment_table = """
        CREATE TABLE IF NOT EXISTS shipment_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            shipment_date DATE,
            area VARCHAR(100),
            product_category VARCHAR(100),
            quantity_sent INT,
            shipment_cost DECIMAL(10, 2),
            transportation_mode VARCHAR(50),
            delivery_time_days DECIMAL(5, 2)
        )
        """
        
        create_sales_table = """
        CREATE TABLE IF NOT EXISTS sales_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            sale_date DATE,
            area VARCHAR(100),
            product_category VARCHAR(100),
            units_sold INT,
            revenue DECIMAL(12, 2),
            customer_segment VARCHAR(50),
            profit_margin DECIMAL(5, 2)
        )
        """
        
        create_inventory_table = """
        CREATE TABLE IF NOT EXISTS inventory_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date DATE,
            area VARCHAR(100),
            product_category VARCHAR(100),
            remaining_stock INT,
            holding_cost DECIMAL(10, 2),
            stockout_count INT,
            days_of_supply INT
        )
        """
        
        create_cost_table = """
        CREATE TABLE IF NOT EXISTS cost_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date DATE,
            area VARCHAR(100),
            transportation_cost DECIMAL(10, 2),
            warehouse_cost DECIMAL(10, 2),
            handling_cost DECIMAL(10, 2),
            other_costs DECIMAL(10, 2)
        )
        """
        
        try:
            if not self.connection and not self.connect():
                return False
                
            self.execute_query(create_shipment_table)
            self.execute_query(create_sales_table)
            self.execute_query(create_inventory_table)
            self.execute_query(create_cost_table)
            return True
        except SQLAlchemyError as e:
            st.error(f"Table creation error: {str(e)}")
            return False
    
    def insert_dataframe(self, df, table_name):
        """
        Insert a pandas DataFrame into a MySQL table.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to insert
        table_name : str
            Target table name
            
        Returns:
        --------
        bool
            True if insertion is successful, False otherwise
        """
        if not self.connection and not self.connect():
            return False
            
        try:
            df.to_sql(name=table_name, con=self.engine, if_exists='append', index=False)
            return True
        except SQLAlchemyError as e:
            st.error(f"Data insertion error: {str(e)}")
            return False
    
    def table_exists(self, table_name):
        """
        Check if a table exists in the database.
        
        Parameters:
        -----------
        table_name : str
            Name of the table to check
            
        Returns:
        --------
        bool
            True if the table exists, False otherwise
        """
        if not self.connection and not self.connect():
            return False
            
        query = f"SHOW TABLES LIKE '{table_name}'"
        result = self.execute_query(query)
        if result is None:
            return False
        try:
            return result.rowcount > 0
        except AttributeError:
            # Alternative approach if rowcount is not available
            try:
                rows = result.fetchall()
                return len(rows) > 0
            except:
                return False
    
    def table_has_data(self, table_name):
        """
        Check if a table has any data.
        
        Parameters:
        -----------
        table_name : str
            Name of the table to check
            
        Returns:
        --------
        bool
            True if the table has data, False otherwise
        """
        if not self.connection and not self.connect():
            return False
        
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        result = self.query_to_dataframe(query)
        return result['count'].iloc[0] > 0
    
    def get_shipment_data(self, area=None, start_date=None, end_date=None):
        """
        Fetch shipment data with optional filtering.
        
        Parameters:
        -----------
        area : str, optional
            Filter by specific area
        start_date : str, optional
            Filter by start date (format: 'YYYY-MM-DD')
        end_date : str, optional
            Filter by end date (format: 'YYYY-MM-DD')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing shipment data
        """
        query = "SELECT * FROM shipment_data WHERE 1=1"
        params = {}
        
        if area:
            query += " AND area = :area"
            params['area'] = area
        
        if start_date:
            query += " AND shipment_date >= :start_date"
            params['start_date'] = start_date
        
        if end_date:
            query += " AND shipment_date <= :end_date"
            params['end_date'] = end_date
        
        return self.query_to_dataframe(query, params)
    
    def get_sales_data(self, area=None, start_date=None, end_date=None):
        """
        Fetch sales data with optional filtering.
        
        Parameters:
        -----------
        area : str, optional
            Filter by specific area
        start_date : str, optional
            Filter by start date (format: 'YYYY-MM-DD')
        end_date : str, optional
            Filter by end date (format: 'YYYY-MM-DD')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing sales data
        """
        query = "SELECT * FROM sales_data WHERE 1=1"
        params = {}
        
        if area:
            query += " AND area = :area"
            params['area'] = area
        
        if start_date:
            query += " AND sale_date >= :start_date"
            params['start_date'] = start_date
        
        if end_date:
            query += " AND sale_date <= :end_date"
            params['end_date'] = end_date
        
        return self.query_to_dataframe(query, params)
    
    def get_inventory_data(self, area=None, start_date=None, end_date=None):
        """
        Fetch inventory data with optional filtering.
        
        Parameters:
        -----------
        area : str, optional
            Filter by specific area
        start_date : str, optional
            Filter by start date (format: 'YYYY-MM-DD')
        end_date : str, optional
            Filter by end date (format: 'YYYY-MM-DD')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing inventory data
        """
        query = "SELECT * FROM inventory_data WHERE 1=1"
        params = {}
        
        if area:
            query += " AND area = :area"
            params['area'] = area
        
        if start_date:
            query += " AND date >= :start_date"
            params['start_date'] = start_date
        
        if end_date:
            query += " AND date <= :end_date"
            params['end_date'] = end_date
        
        return self.query_to_dataframe(query, params)
    
    def get_cost_data(self, area=None, start_date=None, end_date=None):
        """
        Fetch cost data with optional filtering.
        
        Parameters:
        -----------
        area : str, optional
            Filter by specific area
        start_date : str, optional
            Filter by start date (format: 'YYYY-MM-DD')
        end_date : str, optional
            Filter by end date (format: 'YYYY-MM-DD')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing cost data
        """
        query = "SELECT * FROM cost_data WHERE 1=1"
        params = {}
        
        if area:
            query += " AND area = :area"
            params['area'] = area
        
        if start_date:
            query += " AND date >= :start_date"
            params['start_date'] = start_date
        
        if end_date:
            query += " AND date <= :end_date"
            params['end_date'] = end_date
        
        return self.query_to_dataframe(query, params)
    
    def create_database_if_not_exists(self):
        """
        Create the database if it doesn't exist.
        This should be called before connecting to the specific database.
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Connect to MySQL server without specifying a database
            temp_connection_string = f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}"
            temp_engine = create_engine(temp_connection_string)
            
            with temp_engine.connect() as conn:
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {self.database}"))
                
            return True
        except SQLAlchemyError as e:
            st.error(f"Database creation error: {str(e)}")
            return False
            
    def get_sample_data_status(self):
        """
        Check if sample data has been loaded into all required tables.
        
        Returns:
        --------
        bool
            True if all tables exist and have data, False otherwise
        """
        required_tables = ['shipment_data', 'sales_data', 'inventory_data', 'cost_data']
        
        if not self.connection and not self.connect():
            return False
        
        for table in required_tables:
            if not self.table_exists(table) or not self.table_has_data(table):
                return False
        
        return True
    
    def insert_sample_data(self, shipment_df, sales_df, inventory_df, cost_df):
        """
        Insert sample data into the database tables.
        
        Parameters:
        -----------
        shipment_df : pandas.DataFrame
            Sample shipment data
        sales_df : pandas.DataFrame
            Sample sales data
        inventory_df : pandas.DataFrame
            Sample inventory data
        cost_df : pandas.DataFrame
            Sample cost data
            
        Returns:
        --------
        bool
            True if all insertions are successful, False otherwise
        """
        if not self.connection and not self.connect():
            return False
            
        try:
            # Clear existing data if any
            self.execute_query("TRUNCATE TABLE shipment_data")
            self.execute_query("TRUNCATE TABLE sales_data")
            self.execute_query("TRUNCATE TABLE inventory_data")
            self.execute_query("TRUNCATE TABLE cost_data")
            
            # Insert new data
            self.insert_dataframe(shipment_df, 'shipment_data')
            self.insert_dataframe(sales_df, 'sales_data')
            self.insert_dataframe(inventory_df, 'inventory_data')
            self.insert_dataframe(cost_df, 'cost_data')
            
            return True
        except SQLAlchemyError as e:
            st.error(f"Sample data insertion error: {str(e)}")
            return False

def get_db_manager():
    """
    Get a singleton instance of the DatabaseManager.
    This ensures we're using the same connection throughout the app.
    
    Returns:
    --------
    DatabaseManager
        Instance of the DatabaseManager
    """
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
        
        # Create the database and tables if they don't exist
        st.session_state.db_manager.create_database_if_not_exists()
        st.session_state.db_manager.create_tables()
        
    return st.session_state.db_manager