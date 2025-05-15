import streamlit as st
import os
import pandas as pd
from utils.data_processor import DataProcessor
from utils.database import get_db_manager
from utils.db_setup import generate_all_sample_data

st.set_page_config(
    page_title="Supply Chain Optimizer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for data storage
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'shipment_data' not in st.session_state:
    st.session_state.shipment_data = None
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = None
if 'inventory_data' not in st.session_state:
    st.session_state.inventory_data = None
if 'cost_data' not in st.session_state:
    st.session_state.cost_data = None

# App title and description
st.title("AI-Powered Supply Chain Analytics Tool")
st.markdown("""
This tool helps optimize shipments based on sales data and profitability analysis.
Connect to your MySQL database or upload CSV files to get started with analytics, forecasting, and recommendations.
""")

# Display a relevant supply chain image
st.image("https://pixabay.com/get/g71e38b13ab93958b58bd7d985c9fb2fdd30d5261384ed85be808c90b8f193fbe0c31d871c40cf2a9eb74564d97b9e7e546b30f5a91ecd3839c9524866c72bfdf_1280.jpg", 
         caption="Supply Chain Analytics", width=800)

# Data source selection
st.header("Data Source")
data_source = st.radio("Select data source:", ["MySQL Database", "CSV File Upload"])

if data_source == "MySQL Database":
    # Database connection section
    st.subheader("MySQL Database Connection")
    
    # Create three columns for the database connection fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        db_host = st.text_input("Host", "localhost", help="MySQL server hostname (default: localhost for XAMPP)")
        db_user = st.text_input("Username", "root", help="Database username (default: root for XAMPP)")
    
    with col2:
        db_password = st.text_input("Password", "", help="Database password (empty by default for XAMPP)", type="password")
        db_name = st.text_input("Database Name", "supply_chain_analytics", help="Name of the database to use")
    
    with col3:
        db_port = st.number_input("Port", 3306, help="MySQL port (default: 3306)")
        connect_button = st.button("Connect to Database")
    
    if connect_button:
        # Get database manager with custom connection parameters
        db_manager = get_db_manager()
        db_manager.host = db_host
        db_manager.user = db_user
        db_manager.password = db_password
        db_manager.database = db_name
        db_manager.port = db_port
        
        # Create database if it doesn't exist
        if db_manager.create_database_if_not_exists():
            st.success(f"Database '{db_name}' is ready!")
            
            # Create tables if they don't exist
            if db_manager.create_tables():
                st.success("Database tables are set up!")
                
                # Check if sample data exists
                if db_manager.get_sample_data_status():
                    st.info("Sample data is already available in the database.")
                    
                    # Get data from database and load into session
                    shipment_data = db_manager.get_shipment_data()
                    sales_data = db_manager.get_sales_data()
                    inventory_data = db_manager.get_inventory_data()
                    cost_data = db_manager.get_cost_data()
                    
                    # Store the data in session state
                    st.session_state.shipment_data = shipment_data
                    st.session_state.sales_data = sales_data
                    st.session_state.inventory_data = inventory_data
                    st.session_state.cost_data = cost_data
                    
                    # Initialize the data processor
                    st.session_state.data_processor = DataProcessor(
                        shipment_data, sales_data, inventory_data, cost_data
                    )
                    
                    st.success("Data loaded successfully from MySQL database!")
                    st.rerun()
                else:
                    # Offer to load sample data
                    if st.button("Load Sample Data into Database"):
                        # Generate and load sample data
                        shipment_df, sales_df, inventory_df, cost_df = generate_all_sample_data()
                        
                        if db_manager.insert_sample_data(shipment_df, sales_df, inventory_df, cost_df):
                            st.success("Sample data inserted into database successfully!")
                            
                            # Store the data in session state
                            st.session_state.shipment_data = shipment_df
                            st.session_state.sales_data = sales_df
                            st.session_state.inventory_data = inventory_df
                            st.session_state.cost_data = cost_df
                            
                            # Initialize the data processor
                            st.session_state.data_processor = DataProcessor(
                                shipment_df, sales_df, inventory_df, cost_df
                            )
                            
                            st.success("Data loaded successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to insert sample data into database.")
            else:
                st.error("Failed to set up database tables.")
        else:
            st.error(f"Failed to create or connect to database '{db_name}'.")
    
    # Display database connection info
    st.info("""
    **Local MySQL Setup Guide (XAMPP)**:
    1. Install XAMPP if not already installed
    2. Start the MySQL service in XAMPP Control Panel
    3. Use the connection details above (defaults should work with XAMPP)
    4. Click "Connect to Database" to establish a connection
    5. If the database doesn't exist, it will be created automatically
    """)

else:  # CSV File Upload
    # File upload section
    st.subheader("CSV File Upload")
    st.markdown("Upload your CSV files containing shipment, sales, inventory, and cost data.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        shipment_file = st.file_uploader("Upload Shipment Data (CSV)", type=["csv"])
        sales_file = st.file_uploader("Upload Sales Data (CSV)", type=["csv"])
    
    with col2:
        inventory_file = st.file_uploader("Upload Inventory Data (CSV)", type=["csv"])
        cost_file = st.file_uploader("Upload Cost Data (CSV)", type=["csv"])
    
    # Process the uploaded files
    if shipment_file and sales_file and inventory_file and cost_file:
        try:
            # Read the uploaded files into DataFrames
            shipment_data = pd.read_csv(shipment_file)
            sales_data = pd.read_csv(sales_file)
            inventory_data = pd.read_csv(inventory_file)
            cost_data = pd.read_csv(cost_file)
            
            # Store the data in session state
            st.session_state.shipment_data = shipment_data
            st.session_state.sales_data = sales_data
            st.session_state.inventory_data = inventory_data
            st.session_state.cost_data = cost_data
            
            # Initialize the data processor
            st.session_state.data_processor = DataProcessor(
                shipment_data, sales_data, inventory_data, cost_data
            )
            
            st.success("All data files have been successfully uploaded and processed!")
            
            # Show option to save to database
            db_save = st.checkbox("Save this data to MySQL database for future use")
            if db_save:
                # Database connection section for saving
                st.subheader("MySQL Database Connection")
                
                # Create three columns for the database connection fields
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    db_host = st.text_input("Host", "localhost", key="save_host")
                    db_user = st.text_input("Username", "root", key="save_user")
                
                with col2:
                    db_password = st.text_input("Password", "", key="save_pass", type="password")
                    db_name = st.text_input("Database Name", "supply_chain_analytics", key="save_db")
                
                with col3:
                    db_port = st.number_input("Port", 3306, key="save_port")
                    save_button = st.button("Save to Database")
                
                if save_button:
                    # Get database manager with custom connection parameters
                    db_manager = get_db_manager()
                    db_manager.host = db_host
                    db_manager.user = db_user
                    db_manager.password = db_password
                    db_manager.database = db_name
                    db_manager.port = db_port
                    
                    # Create database and tables if they don't exist
                    if db_manager.create_database_if_not_exists() and db_manager.create_tables():
                        # Insert the data into database
                        if db_manager.insert_sample_data(shipment_data, sales_data, inventory_data, cost_data):
                            st.success("Data saved to database successfully!")
                        else:
                            st.error("Failed to save data to database.")
                    else:
                        st.error("Failed to create database or tables.")
            
            # Show data preview and summary
            st.header("Data Preview")
            tabs = st.tabs(["Shipment Data", "Sales Data", "Inventory Data", "Cost Data"])
            
            with tabs[0]:
                st.subheader("Shipment Data")
                st.dataframe(shipment_data.head())
                st.write(f"Shape: {shipment_data.shape}")
            
            with tabs[1]:
                st.subheader("Sales Data")
                st.dataframe(sales_data.head())
                st.write(f"Shape: {sales_data.shape}")
            
            with tabs[2]:
                st.subheader("Inventory Data")
                st.dataframe(inventory_data.head())
                st.write(f"Shape: {inventory_data.shape}")
            
            with tabs[3]:
                st.subheader("Cost Data")
                st.dataframe(cost_data.head())
                st.write(f"Shape: {cost_data.shape}")
            
            # Display navigation options
            st.header("Navigate to Analysis Modules")
            st.markdown("""
            Now that your data is uploaded, you can navigate to different analysis modules using the sidebar.
            """)
            
        except Exception as e:
            st.error(f"Error processing the uploaded files: {e}")
            
    else:
        # If files aren't uploaded, show a message with sample data option
        st.info("Please upload all required files to proceed with analysis.")
        
        if st.button("Use Sample Data"):
            try:
                # Generate sample data
                shipment_data, sales_data, inventory_data, cost_data = generate_all_sample_data()
                
                # Store the data in session state
                st.session_state.shipment_data = shipment_data
                st.session_state.sales_data = sales_data
                st.session_state.inventory_data = inventory_data
                st.session_state.cost_data = cost_data
                
                # Initialize the data processor
                st.session_state.data_processor = DataProcessor(
                    shipment_data, sales_data, inventory_data, cost_data
                )
                
                st.success("Sample data generated successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error generating sample data: {e}")

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("### Analysis Modules")

# Only show navigation options if data is loaded
if st.session_state.data_processor is not None:
    # Add navigation options to other pages in the pages directory
    st.sidebar.page_link("pages/dashboard.py", label="Main Dashboard")
    st.sidebar.page_link("pages/area_analysis.py", label="Area Performance Analysis")
    st.sidebar.page_link("pages/forecasting.py", label="Demand Forecasting")
    st.sidebar.page_link("pages/recommendations.py", label="Shipment Recommendations")
    st.sidebar.page_link("pages/geo_visualization.py", label="Geo-Visualization")
    
    # Add new advanced analytics page
    st.sidebar.markdown("### Advanced Features")
    st.sidebar.page_link("pages/advanced_analytics_page.py", label="Advanced Analytics", icon="📊")
    
    # Settings and about section
    st.sidebar.markdown("---")
    st.sidebar.page_link("pages/settings.py", label="Settings", icon="⚙️")
    
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This application uses machine learning to optimize supply chain operations "
        "by analyzing historical data and providing actionable insights."
    )
