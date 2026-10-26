import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# Find the main project directory automatically
BASE_DIR = Path(__file__).resolve().parent.parent

# Define file paths
DATA_PATH = BASE_DIR / "data" / "prepared_support_tickets.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Create folders if they do not exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Check whether the prepared dataset exists
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Prepared dataset not found: {DATA_PATH}\n"
        "Run prepare_data.py from the project directory first."
    )

# Load the correct prepared dataset
df = pd.read_csv(DATA_PATH)

print("Prepared dataset loaded successfully!")
print("Dataset path:", DATA_PATH)
print("Dataset shape:", df.shape)
print("Available columns:", df.columns.tolist())

# Check required columns
required_columns = [
    "Ticket Text",
    "Ticket Type",
    "Ticket Priority"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        "Please run prepare_data.py again."
    )

# Prepare input text
X = df["Ticket Text"].fillna("").astype(str)

# Define the two prediction tasks
targets = {
    "category": "Ticket Type",
    "priority": "Ticket Priority"
}

results = []

# Train and evaluate each model
for model_name, target_column in targets.items():

    print("\n" + "=" * 60)
    print(f"TRAINING {model_name.upper()} MODEL")
    print("=" * 60)

    y = df[target_column]

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Build the text classification pipeline
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=20000,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    # Train the model
    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Predict labels for unseen test examples
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro F1-score: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Save the trained model
    model_path = MODELS_DIR / f"ticket_{model_name}_model.joblib"
    joblib.dump(model, model_path)

    print("Model saved to:", model_path)

    # Generate confusion matrix
    labels = model.named_steps["classifier"].classes_

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title(f"{model_name.title()} Model - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    image_path = OUTPUTS_DIR / f"{model_name}_confusion_matrix.png"

    plt.savefig(image_path, dpi=150)
    plt.close()

    print("Confusion matrix saved to:", image_path)

    # Store evaluation results
    results.append({
        "Model": model_name.title(),
        "Target": target_column,
        "Accuracy": round(accuracy, 4),
        "Macro F1-score": round(macro_f1, 4),
        "Training Samples": len(X_train),
        "Testing Samples": len(X_test)
    })

# Save the evaluation summary
results_df = pd.DataFrame(results)

metrics_path = OUTPUTS_DIR / "model_metrics.csv"
results_df.to_csv(metrics_path, index=False)

print("\n" + "=" * 60)
print("FINAL MODEL SUMMARY")
print("=" * 60)

print(results_df.to_string(index=False))

print("\nMetrics saved to:", metrics_path)
print("\nBoth models trained and evaluated successfully!")