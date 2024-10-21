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

    return restaurants_df, customers_df

# Step 2: Cluster restaurants and customers to allocate collectors and deliverers
@st.cache_data
def cluster_data(restaurants_df, customers_df, num_collectors, num_deliverers):
    kmeans_collectors = KMeans(n_clusters=num_collectors, random_state=0)
    collector_clusters = kmeans_collectors.fit(restaurants_df[['Grid_X', 'Grid_Y']])
    collectors_df = pd.DataFrame(collector_clusters.cluster_centers_, columns=['Grid_X', 'Grid_Y'])
    collectors_df['Collector_ID'] = [f'Collector_{i+1}' for i in range(num_collectors)]

    kmeans_deliverers = KMeans(n_clusters=num_deliverers, random_state=0)
    deliverer_clusters = kmeans_deliverers.fit(customers_df[['Grid_X', 'Grid_Y']])
    deliverers_df = pd.DataFrame(deliverer_clusters.cluster_centers_, columns=['Grid_X', 'Grid_Y'])
    deliverers_df['Deliverer_ID'] = [f'Deliverer_{i+1}' for i in range(num_deliverers)]

    return collectors_df, deliverers_df

# Step 3: Create a map to visualize the data
def create_map(restaurants_df, customers_df, collectors_df, deliverers_df):
    m = folium.Map(location=[PUNE_LAT, PUNE_LON], zoom_start=13)

    # Add restaurant markers
    for _, row in restaurants_df.iterrows():
        folium.Marker([row['Grid_X'], row['Grid_Y']], popup=f"Restaurant {row['Restaurant_ID']}",
                      icon=folium.Icon(color="blue", icon="cutlery")).add_to(m)

    # Add customer markers
    for _, row in customers_df.iterrows():
        folium.Marker([row['Grid_X'], row['Grid_Y']], popup=f"Customer {row['Customer_ID']}",
                      icon=folium.Icon(color="green", icon="user")).add_to(m)

    # Add collector markers
    for _, row in collectors_df.iterrows():
        folium.Marker([row['Grid_X'], row['Grid_Y']], popup=f"Collector {row['Collector_ID']}",
                      icon=folium.Icon(color="red", icon="wrench")).add_to(m)

    # Add deliverer markers
    for _, row in deliverers_df.iterrows():
        folium.Marker([row['Grid_X'], row['Grid_Y']], popup=f"Deliverer {row['Deliverer_ID']}",
                      icon=folium.Icon(color="purple", icon="car")).add_to(m)

    return m

# Step 4: Streamlit app
def main():
    st.title("Delivery Efficiency Simulation in Pune, Maharashtra")

    # Step 1: User input for number of restaurants, customers, and drivers
    num_restaurants = st.number_input("Number of Restaurants", min_value=1, value=50, step=1)
    num_customers = st.number_input("Number of Customers", min_value=1, value=50, step=1)
    num_drivers = st.number_input("Number of Drivers", min_value=1, value=20, step=1)

    num_collectors = int(num_drivers * COLLECTOR_RATIO)
    num_deliverers = int(num_drivers * DELIVERER_RATIO)

    # Generate data
    restaurants_df, customers_df = generate_data(num_restaurants, num_customers)
    st.write(f"Generated {num_restaurants} restaurants and {num_customers} customers.")

    # Cluster and allocate drivers
    collectors_df, deliverers_df = cluster_data(restaurants_df, customers_df, num_collectors, num_deliverers)
    st.write(f"Allocated {num_collectors} collectors and {num_deliverers} deliverers.")

    # Visualize the result on a map
    map_obj = create_map(restaurants_df, customers_df, collectors_df, deliverers_df)

    # Display the map
    st_folium(map_obj, width=700, height=500)

if __name__ == "__main__":
    main()
