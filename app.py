import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Support Ticket Intelligence",
    page_icon="🎫",
    layout="wide"
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "prepared_support_tickets.csv"
CATEGORY_MODEL_PATH = BASE_DIR / "models" / "ticket_category_model.joblib"
PRIORITY_MODEL_PATH = BASE_DIR / "models" / "ticket_priority_model.joblib"
METRICS_PATH = BASE_DIR / "outputs" / "model_metrics.csv"

CATEGORY_MATRIX_PATH = BASE_DIR / "outputs" / "category_confusion_matrix.png"
PRIORITY_MATRIX_PATH = BASE_DIR / "outputs" / "priority_confusion_matrix.png"

# --------------------------------------------------
# LOAD DATA AND MODELS
# --------------------------------------------------

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    category_model = joblib.load(CATEGORY_MODEL_PATH)
    priority_model = joblib.load(PRIORITY_MODEL_PATH)
    return category_model, priority_model

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return pd.DataFrame()

# Check required files
required_files = [
    DATA_PATH,
    CATEGORY_MODEL_PATH,
    PRIORITY_MODEL_PATH
]

missing_files = [str(path) for path in required_files if not path.exists()]

if missing_files:
    st.error("Some required project files are missing:")
    for file in missing_files:
        st.write(f"- {file}")
    st.info("Run the data preparation and model training scripts first.")
    st.stop()

df = load_dataset()
category_model, priority_model = load_models()
metrics_df = load_metrics()

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🎫 Ticket Intelligence")
st.sidebar.caption("Support Ticket Classification")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Predict Ticket",
        "Dataset Explorer",
        "Model Performance"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Machine Learning Project")
st.sidebar.caption("TF-IDF + Logistic Regression")

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.title("🎫 Support Ticket Intelligence")
    st.write(
        "A machine learning dashboard for classifying customer support "
        "tickets and estimating their priority."
    )

    st.warning(
        "Prototype notice: This application uses a synthetic practice "
        "dataset. Predictions should be reviewed by a human and must not "
        "be treated as verified real-world classifications."
    )

    total_tickets = len(df)
    total_categories = df["Ticket Type"].nunique()
    total_priorities = df["Ticket Priority"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Tickets", f"{total_tickets:,}")
    col2.metric("Ticket Categories", total_categories)
    col3.metric("Priority Levels", total_priorities)
    col4.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Tickets by Category")

        category_counts = df["Ticket Type"].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))
        category_counts.plot(kind="bar", ax=ax)

        ax.set_xlabel("Ticket Category")
        ax.set_ylabel("Number of Tickets")
        ax.tick_params(axis="x", rotation=35)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Tickets by Priority")

        priority_counts = df["Ticket Priority"].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))
        priority_counts.plot(kind="bar", ax=ax)

        ax.set_xlabel("Ticket Priority")
        ax.set_ylabel("Number of Tickets")
        ax.tick_params(axis="x", rotation=0)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Recent Ticket Records")
    st.dataframe(
        df[
            [
                "Ticket ID",
                "Ticket Subject",
                "Ticket Type",
                "Ticket Priority",
                "Ticket Channel"
            ]
        ].head(10),
        use_container_width=True
    )

# --------------------------------------------------
# PREDICT TICKET
# --------------------------------------------------

elif page == "Predict Ticket":

    st.title("🔍 Predict a Support Ticket")
    st.write(
        "Enter the customer's ticket subject and description to generate "
        "predicted category and priority labels."
    )

    with st.form("ticket_prediction_form"):

        subject = st.text_input(
            "Ticket Subject",
            placeholder="Example: Refund has not arrived"
        )

        description = st.text_area(
            "Ticket Description",
            placeholder="Describe the customer's issue...",
            height=180
        )

        submitted = st.form_submit_button(
            "Classify Ticket",
            use_container_width=True
        )

    if submitted:

        if not subject.strip() or not description.strip():
            st.error("Please enter both a subject and a description.")

        else:
            ticket_text = subject.strip() + " " + description.strip()

            category_prediction = category_model.predict([ticket_text])[0]
            priority_prediction = priority_model.predict([ticket_text])[0]

            category_probabilities = category_model.predict_proba(
                [ticket_text]
            )[0]

            priority_probabilities = priority_model.predict_proba(
                [ticket_text]
            )[0]

            category_score = float(max(category_probabilities) * 100)
            priority_score = float(max(priority_probabilities) * 100)

            st.divider()
            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Predicted Category")
                st.success(category_prediction)
                st.progress(min(category_score / 100, 1.0))
                st.caption(
                    f"Model score: {category_score:.2f}%"
                )

            with col2:
                st.markdown("### Predicted Priority")
                st.info(priority_prediction)
                st.progress(min(priority_score / 100, 1.0))
                st.caption(
                    f"Model score: {priority_score:.2f}%"
                )

            st.subheader("Ticket Summary")

            result_df = pd.DataFrame([{
                "Ticket Subject": subject,
                "Ticket Description": description,
                "Predicted Category": category_prediction,
                "Predicted Priority": priority_prediction,
                "Category Model Score (%)": round(category_score, 2),
                "Priority Model Score (%)": round(priority_score, 2)
            }])

            st.dataframe(result_df, use_container_width=True)

            st.download_button(
                "Download Prediction as CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="ticket_prediction.csv",
                mime="text/csv"
            )

            st.warning(
                "These scores are model confidence estimates, not "
                "guarantees of correctness. Review the prediction before "
                "using it to make support decisions."
            )

# --------------------------------------------------
# DATASET EXPLORER
# --------------------------------------------------

elif page == "Dataset Explorer":

    st.title("📊 Dataset Explorer")
    st.write("Explore the support-ticket practice dataset.")

    col1, col2 = st.columns(2)

    with col1:
        selected_category = st.selectbox(
            "Filter by Category",
            ["All"] + sorted(df["Ticket Type"].dropna().unique().tolist())
        )

    with col2:
        selected_priority = st.selectbox(
            "Filter by Priority",
            ["All"] + sorted(df["Ticket Priority"].dropna().unique().tolist())
        )

    filtered_df = df.copy()

    if selected_category != "All":
        filtered_df = filtered_df[
            filtered_df["Ticket Type"] == selected_category
        ]

    if selected_priority != "All":
        filtered_df = filtered_df[
            filtered_df["Ticket Priority"] == selected_priority
        ]

    st.metric("Matching Tickets", len(filtered_df))

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Filtered Dataset",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_support_tickets.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("📈 Model Performance")
    st.write(
        "Evaluation results from the saved training run."
    )

    if not metrics_df.empty:

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "These metrics describe performance on the held-out split "
            "used during training. They do not guarantee performance "
            "on new or real-world tickets."
        )

    else:
        st.info("Model metrics have not been generated yet.")

    st.divider()
    st.subheader("Confusion Matrices")

    left, right = st.columns(2)

    with left:
        st.markdown("**Ticket Category Model**")

        if CATEGORY_MATRIX_PATH.exists():
            st.image(
                str(CATEGORY_MATRIX_PATH),
                use_container_width=True
            )
        else:
            st.info("Category confusion matrix not found.")

    with right:
        st.markdown("**Ticket Priority Model**")

        if PRIORITY_MATRIX_PATH.exists():
            st.image(
                str(PRIORITY_MATRIX_PATH),
                use_container_width=True
            )
        else:
            st.info("Priority confusion matrix not found.")

    st.warning(
        "Both models previously achieved 100% on the current test split. "
        "Because the dataset is synthetic and the score is unusually high, "
        "validate the models on independently created tickets before "
        "claiming that they are reliable."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.divider()
st.sidebar.caption("Built with Python, scikit-learn and Streamlit")