import random
import pandas as pd

# Define the initial data
initial_data = [
    (1, '2024-01-12', 'Tesla', 8, 68000),
    (2, '2024-09-10', 'Humane pin', 9, 650),
    (3, '2024-04-08', 'iPhone', 3, 800),
    (4, '2024-03-03', 'Humane pin', 8, 800),
    (5, '2024-08-20', 'Tesla', 3, 71000)
]

# Extract unique values for each column
dates = pd.date_range(start="1995-01-01", end="2024-12-31", freq='D')
products = ["iPhone", "Tesla", "Humane pin"]
employee_ids = [record[3] for record in initial_data]
prices = {
    "iPhone": range(600, 1200, 50),
    "Tesla": range(50000, 80000, 1000),
    "Humane pin": range(400, 1000, 50)
}

# Generate 100 random records
records = []
for i in range(1, 1000000):
    date = random.choice(dates).strftime('%Y-%m-%d')
    product = random.choice(products)
    employee_id = random.choice(employee_ids)
    price = random.choice(prices[product])
    records.append((i, date, product, employee_id, price))

# Convert to DataFrame for better visualization
df = pd.DataFrame(records, columns=['ID', 'Date', 'Product', 'Employee ID', 'Price'])

# Save to CSV file
file_path = 'random_100_records.csv'
df.to_csv(file_path, index=False, header=False)

print(f"100 random records generated and saved to {file_path}")
