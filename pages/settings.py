import streamlit as st
import os
import shutil

# Page configuration
st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

# Page title and description
st.title("Settings")
st.markdown("""
Configure application settings including theme, appearance, and system preferences.
""")

# Theme settings
st.header("Theme Settings")

# Get current theme settings
config_path = ".streamlit/config.toml"
dark_mode_enabled = False

try:
    with open(config_path, "r") as f:
        config_lines = f.readlines()
        
    # Check if dark mode is currently enabled
    for i, line in enumerate(config_lines):
        if line.strip() == "backgroundColor = \"#0E1117\"":
            dark_mode_enabled = True
            break
except:
    st.warning("Unable to read current theme settings.")

# Theme selection
theme_mode = st.radio(
    "Select Theme Mode",
    ["Light Mode", "Dark Mode"],
    index=1 if dark_mode_enabled else 0
)

if st.button("Apply Theme"):
    # Read the current config
    try:
        with open(config_path, "r") as f:
            config_content = f.read()
        
        # Create backup
        backup_path = f"{config_path}.bak"
        with open(backup_path, "w") as f:
            f.write(config_content)
        
        # Modify the theme section based on selection
        if theme_mode == "Dark Mode":
            # Enable dark mode
            new_config = config_content.replace(
                "# Light mode settings (default)\nprimaryColor = \"#4CAF50\"\nbackgroundColor = \"#FFFFFF\"\nsecondaryBackgroundColor = \"#F0F2F6\"\ntextColor = \"#262730\"\n\n# Uncomment these for dark mode\n# primaryColor = \"#4CAF50\"\n# backgroundColor = \"#0E1117\"\n# secondaryBackgroundColor = \"#1E2530\"\n# textColor = \"#FAFAFA\"",
                "# Light mode settings (commented out)\n# primaryColor = \"#4CAF50\"\n# backgroundColor = \"#FFFFFF\"\n# secondaryBackgroundColor = \"#F0F2F6\"\n# textColor = \"#262730\"\n\n# Dark mode settings (active)\nprimaryColor = \"#4CAF50\"\nbackgroundColor = \"#0E1117\"\nsecondaryBackgroundColor = \"#1E2530\"\ntextColor = \"#FAFAFA\""
            )
        else:
            # Enable light mode
            new_config = config_content.replace(
                "# Light mode settings (commented out)\n# primaryColor = \"#4CAF50\"\n# backgroundColor = \"#FFFFFF\"\n# secondaryBackgroundColor = \"#F0F2F6\"\n# textColor = \"#262730\"\n\n# Dark mode settings (active)\nprimaryColor = \"#4CAF50\"\nbackgroundColor = \"#0E1117\"\nsecondaryBackgroundColor = \"#1E2530\"\ntextColor = \"#FAFAFA\"",
                "# Light mode settings (default)\nprimaryColor = \"#4CAF50\"\nbackgroundColor = \"#FFFFFF\"\nsecondaryBackgroundColor = \"#F0F2F6\"\ntextColor = \"#262730\"\n\n# Uncomment these for dark mode\n# primaryColor = \"#4CAF50\"\n# backgroundColor = \"#0E1117\"\n# secondaryBackgroundColor = \"#1E2530\"\n# textColor = \"#FAFAFA\""
            )
            
        # Write the updated config
        with open(config_path, "w") as f:
            f.write(new_config)
            
        st.success(f"{theme_mode} applied successfully. Please refresh the page to see the changes.")
        
        # Provide a button to restart the application
        if st.button("Restart Application"):
            st.rerun()
            
    except Exception as e:
        st.error(f"Error applying theme: {str(e)}")

# Application Settings
st.header("Application Settings")

# Create columns for settings
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Settings")
    sample_data_option = st.checkbox("Load sample data on startup", value=True)
    cache_option = st.checkbox("Enable data caching", value=True)
    
    # Save data settings if changed
    if st.button("Save Data Settings"):
        # In a real application, these would be saved to a configuration file
        st.session_state.sample_data_startup = sample_data_option
        st.session_state.enable_caching = cache_option
        st.success("Data settings saved successfully.")

with col2:
    st.subheader("Display Settings")
    
    decimal_places = st.slider("Decimal places for metrics", 0, 4, 2)
    chart_height = st.slider("Default chart height", 300, 800, 500)
    
    # Save display settings if changed
    if st.button("Save Display Settings"):
        # In a real application, these would be saved to a configuration file
        st.session_state.decimal_places = decimal_places
        st.session_state.chart_height = chart_height
        st.success("Display settings saved successfully.")

# API Settings
st.header("API Connection Settings")

# Setup API connection settings
api_enabled = st.checkbox("Enable API Connections", value=False)

if api_enabled:
    api_type = st.selectbox(
        "API Type",
        ["SAP", "Oracle", "Generic REST API"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        api_url = st.text_input("API Base URL", placeholder="https://api.example.com/v1")
        api_username = st.text_input("Username (if required)")
    
    with col2:
        api_key = st.text_input("API Key (if required)", type="password")
        api_password = st.text_input("Password (if required)", type="password")
    
    # Save API settings
    if st.button("Save API Settings"):
        # In a real application, these would be securely stored
        st.session_state.api_enabled = api_enabled
        st.session_state.api_type = api_type
        st.session_state.api_url = api_url
        st.session_state.api_username = api_username
        st.session_state.api_key = api_key
        st.session_state.api_password = api_password
        
        st.success("API settings saved successfully.")

# About section
st.header("About")
st.markdown("""
### Supply Chain Analytics Tool

Version: 1.0.0

This application was built to help organizations optimize their supply chain 
operations through advanced analytics, including inventory optimization, 
route planning, risk assessment, and forecasting.

**Features:**
- Data visualization and analytics
- Inventory optimization
- Route optimization
- Risk assessment
- Advanced forecasting
- What-if scenario analysis
- Reporting and exports

For more information, please refer to the project documentation.
""")

# Footer
st.markdown("---")
st.markdown("© 2025 Supply Chain Analytics. All rights reserved.")