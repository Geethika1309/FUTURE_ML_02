import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "data/support_ticket_working_dataset.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)

# Combine ticket subject and description
df["Ticket Text"] = (
    df["Ticket Subject"].fillna("").astype(str)
    + " "
    + df["Ticket Description"].fillna("").astype(str)
)

# Remove unnecessary spaces
df["Ticket Text"] = df["Ticket Text"].str.strip()

# Remove rows with empty text or missing target labels
df = df[
    (df["Ticket Text"] != "")
    & df["Ticket Type"].notna()
    & df["Ticket Priority"].notna()
].copy()

# Remove duplicate text entries to reduce repeated examples
df = df.drop_duplicates(subset=["Ticket Text"]).reset_index(drop=True)

print("\nPrepared dataset shape:", df.shape)

print("\nTicket categories:")
print(df["Ticket Type"].value_counts())

print("\nTicket priorities:")
print(df["Ticket Priority"].value_counts())

# Save the prepared dataset
output_path = "data/prepared_support_tickets.csv"
df.to_csv(output_path, index=False)

print(f"\nPrepared dataset saved to: {output_path}")

# Split data for category prediction
X_category = df["Ticket Text"]
y_category = df["Ticket Type"]

X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(
    X_category,
    y_category,
    test_size=0.2,
    random_state=42,
    stratify=y_category
)

print("\nCategory classification split:")
print("Training examples:", len(X_cat_train))
print("Testing examples:", len(X_cat_test))

# Split data for priority prediction
X_priority = df["Ticket Text"]
y_priority = df["Ticket Priority"]

X_pri_train, X_pri_test, y_pri_train, y_pri_test = train_test_split(
    X_priority,
    y_priority,
    test_size=0.2,
    random_state=42,
    stratify=y_priority
)

print("\nPriority classification split:")
print("Training examples:", len(X_pri_train))
print("Testing examples:", len(X_pri_test))

print("\nData preparation completed successfully!")