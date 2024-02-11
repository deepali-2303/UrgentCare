import sqlite3
import random

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('hospital_database.db')

# Create a cursor object to interact with the database
cursor = conn.cursor()

# Create a table to store hospital information
cursor.execute('''CREATE TABLE IF NOT EXISTS hospitals
                (name TEXT, latitude REAL, longitude REAL)''')

# List of hospitals with their latitude and longitude coordinates
hospitals = [
    ("Hospital A", 40.7128, -74.0060),
    ("Hospital B", 34.0522, -118.2437),
    ("Hospital C", 19,72),
    ("Hospital D", 20.7604, 75.3698),
    ("Hospital E", 33.4484, -112.0740)
]

# Insert hospitals data into the table
cursor.executemany("INSERT INTO hospitals (name, latitude, longitude) VALUES (?, ?, ?)", hospitals)

# Add some random values intentionally
for _ in range(5):
    random_name = f"Hospital {random.choice(['F', 'G', 'H', 'I', 'J'])}"
    random_lat = random.uniform(-90, 90)
    random_long = random.uniform(-180, 180)
    cursor.execute("INSERT INTO hospitals (name, latitude, longitude) VALUES (?, ?, ?)", (random_name, random_lat, random_long))

# Commit changes and close connection
conn.commit()
conn.close()

print("Database created and populated successfully.")
