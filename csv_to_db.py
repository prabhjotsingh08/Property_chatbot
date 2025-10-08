import sqlite3
import pandas as pd

# Load CSV file into pandas DataFrame
csv_file = "clean_data.csv"  # change this to your CSV filename
df = pd.read_csv(csv_file)

# Connect to SQLite database (creates one if it doesn't exist)
conn = sqlite3.connect("real_estate.db")
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    location TEXT,
    total_sqft REAL,
    bath INTEGER,
    price REAL,
    bhk INTEGER
);
""")

# Insert CSV data into table
df.to_sql("properties", conn, if_exists="append", index=False)

# Commit and close
conn.commit()
conn.close()

print("✅ Data inserted into database 'real_estate.db' successfully!")
