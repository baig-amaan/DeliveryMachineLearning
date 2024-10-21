import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium

# Constants
GRID_SIZE = 0.05  # Adjust the size of the area in degrees (approx. 5 km)
COLLECTOR_RATIO = 0.4
DELIVERER_RATIO = 0.6

# Pune Coordinates
PUNE_LAT, PUNE_LON = 18.5204, 73.8567

# Helper function to create random datasets around a central point
def create_random_dataset(num_points, center_lat, center_lon, grid_size):
    return pd.DataFrame({
        'Grid_X': np.random.uniform(center_lat - grid_size, center_lat + grid_size, num_points),
        'Grid_Y': np.random.uniform(center_lon - grid_size, center_lon + grid_size, num_points)
    })

# Step 1: Generate Random Restaurant and Customer Locations
@st.cache_data
def generate_data(num_restaurants, num_customers):
    restaurants_df = create_random_dataset(num_restaurants, PUNE_LAT, PUNE_LON, GRID_SIZE)
    restaurants_df['Restaurant_ID'] = [f'R{i+1}' for i in range(num_restaurants)]

    customers_df = create_random_dataset(num_customers, PUNE_LAT, PUNE_LON, GRID_SIZE)
    customers_df['Customer_ID'] = [f'C{i+1}' for i in range(num_customers)]

    return
