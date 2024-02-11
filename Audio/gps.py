import folium
import requests
import sqlite3

def get_location():
    response = requests.get('https://ipinfo.io/json')
    data = response.json()
    city = data.get('city')
    region = data.get('region')
    country = data.get('country')
    loc = data.get('loc')
    if loc:
        latitude, longitude = loc.split(',')
    else:
        latitude = longitude = None
    return city, region, country, latitude, longitude

def connect_to_database():
    # Connect to your SQLite database
    conn = sqlite3.connect('hospital_database.db')
    return conn

# Get location information
city, region, country, latitude, longitude = get_location()

# Create a map centered at the obtained latitude and longitude
map = folium.Map(location=[float(latitude), float(longitude)], zoom_start=10)

# Add a marker for the obtained location
folium.Marker([float(latitude), float(longitude)], popup=f"{city}, {region}, {country}").add_to(map)

# Connect to the SQLite database
conn = connect_to_database()

# Get hospitals from the database
cur = conn.cursor()
cur.execute("SELECT name, latitude, longitude FROM hospitals;")
map_center = [float(latitude), float(longitude)]


hospitals = cur.fetchall()

# Extract hospital coordinates for FastMarkerCluster
hospital_coordinates = [[float(latitude), float(longitude)] for name, latitude, longitude in hospitals]

# Add FastMarkerCluster to the map
marker_cluster = FastMarkerCluster(hospital_coordinates, name="Hospitals").add_to(map_center)

# Add markers for all hospitals to the cluster
for hospital in hospitals:
    name, hospital_lat, hospital_lon = hospital
    folium.Marker(location=[hospital_lat, hospital_lon], popup=name).add_to(marker_cluster)

# # Save the map to an HTML file
# my_map.save("map_with_hospitals.html")

# # Close the database connection
# conn.close()

# Add markers for all hospitals
for hospital in hospitals:
    name, hospital_lat, hospital_lon = hospital
    folium.Marker(location=[hospital_lat, hospital_lon], popup=name).add_to(map)

my_map = folium.Map(location=map_center, zoom_start=10)

from folium.plugins import FastMarkerCluster

# Add FastMarkerCluster to the map
marker_cluster = FastMarkerCluster()
marker_cluster.add_to(my_map)
# Save the map to an HTML file
map.save("map_with_hospitals.html")

# Close the database connection
conn.close()
