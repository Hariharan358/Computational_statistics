import pandas as pd
import plotly.express as px

# Load the Iris dataset
df = px.data.iris()

# Print the first 10 records
print("1. Print the first 10 records:")
print(df.head(10))

# Print the total number of rows and columns
num_rows, num_columns = df.shape
print("\n2. Print the total number of rows and columns:")
print("(", num_rows, ",", num_columns, ")")

# Print the column names
print("\n3. Print the column names [Attribute Names] of the dataset:")
print(df.columns.tolist())

# Find the mean of all numerical attributes
mean_values = df.mean(numeric_only=True)
print("\n4. Mean of all numerical attributes with 15 decimal places:")
for column, mean in mean_values.items():
    print(f"{column} - Mean = {mean:.15f}")
