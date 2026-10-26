import joblib
from pathlib import Path

# Find the main project folder
BASE_DIR = Path(__file__).resolve().parent

# Load the saved models
category_path = BASE_DIR / "models" / "ticket_category_model.joblib"
priority_path = BASE_DIR / "models" / "ticket_priority_model.joblib"

category_model = joblib.load(category_path)
priority_model = joblib.load(priority_path)

print("=" * 55)
print("SUPPORT TICKET CLASSIFICATION SYSTEM")
print("=" * 55)

# Enter a new support ticket
subject = input("\nEnter ticket subject: ").strip()
description = input("Enter ticket description: ").strip()

if not subject or not description:
    print("\nError: Please enter both subject and description.")
else:
    # Combine subject and description
    ticket_text = subject + " " + description

    # Predict category and priority
    category = category_model.predict([ticket_text])[0]
    priority = priority_model.predict([ticket_text])[0]

    # Get the model's predicted class probabilities
    category_probabilities = category_model.predict_proba([ticket_text])[0]
    priority_probabilities = priority_model.predict_proba([ticket_text])[0]

    category_confidence = max(category_probabilities) * 100
    priority_confidence = max(priority_probabilities) * 100

    # Display results
    print("\n" + "=" * 55)
    print("PREDICTION RESULTS")
    print("=" * 55)

    print("Ticket Subject:", subject)
    print("Predicted Category:", category)
    print("Predicted Priority:", priority)

    print(f"Category Model Score: {category_confidence:.2f}%")
    print(f"Priority Model Score: {priority_confidence:.2f}%")

    print("\nNote:")
    print("These percentages are model confidence scores,")
    print("not guarantees that the predictions are correct.")
    print("Please review predictions before taking action.")