import pandas as pd

# Load the dataset
file_path = "data/support_ticket_working_dataset.csv"
df = pd.read_csv(file_path)

# Display basic information
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTicket category distribution:")
print(df["Ticket Type"].value_counts())

print("\nTicket priority distribution:")
print(df["Ticket Priority"].value_counts())