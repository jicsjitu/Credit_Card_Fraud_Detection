import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
import datetime

# Cache the data loading function for faster loading
@st.cache_data
def load_data():
    data = pd.read_csv('creditcard.csv')
    return data

# Load the data
data = load_data()

# Separate legitimate and fraudulent transactions
legit = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]

legit_new = legit.sample(n=len(fraud), random_state=2)
new_df = pd.concat([legit_new, fraud], axis=0)

# Splitting the data into X and Y
X = new_df.drop('Class', axis=1)
Y = new_df['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Checking model performance
train_data_acc = accuracy_score(model.predict(X_train), Y_train)
test_data_acc = accuracy_score(model.predict(X_test), Y_test)
test_precision = precision_score(model.predict(X_test), Y_test)
test_recall = recall_score(model.predict(X_test), Y_test)
test_f1 = f1_score(model.predict(X_test), Y_test)

# Streamlit app
st.title('💳 Credit Card Fraud Detection')
st.write("""
    This app uses a Logistic Regression model to detect fraudulent credit card transactions.
The model is trained on a balanced dataset with equal numbers of legitimate and fraudulent transactions.
Enter the transaction details to predict if it's fraudulent or legitimate.

""")

st.sidebar.title("🛠 App Options")
if st.sidebar.button("Reload Data"):
    st.cache_data.clear()
    st.experimental_rerun()

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #f0f2f6;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput, .stTextArea {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("Model Performance")
st.sidebar.write(f"Training Accuracy: {train_data_acc:.2f}")
st.sidebar.write(f"Testing Accuracy: {test_data_acc:.2f}")
st.sidebar.write(f"Precision: {test_precision:.2f}")
st.sidebar.write(f"Recall: {test_recall:.2f}")
st.sidebar.write(f"F1 Score: {test_f1:.2f}")

st.sidebar.markdown("### About")
st.sidebar.info("""
Name: Jitu Kumar
                 
Linkedln : [Jitu Kumar](https://www.linkedin.com/in/jicsjitu/)
                 
GitHub: [Jitu Kumar](https://github.com/jicsjitu)
                 
Blog : [Medium](https://medium.com/@jicsjitu)
                 
Email: jitukumar9387@gmail.com
                 
""")

# Adding an image
st.sidebar.image("jitu.png", caption="Developer", use_column_width=True)

# Adding a chart for visual feedback
st.sidebar.header("Data Distribution")
if st.sidebar.checkbox("Show Data Distribution"):
    st.sidebar.write("Distribution of Legitimate and Fraudulent Transactions")
    dist_df = pd.DataFrame({'Class': ['Legitimate', 'Fraudulent'], 'Count': [len(legit), len(fraud)]})
    st.sidebar.bar_chart(dist_df.set_index('Class'))

# Initialize session state
if 'input_data' not in st.session_state:
    st.session_state.input_data = ""

# Monitoring functions
def log_prediction(input_data, prediction):
    with open("predictions_log.csv", "a") as log_file:
        log_file.write(f"{datetime.datetime.now()},{input_data},{prediction}\n")

def display_model_metrics():
    st.sidebar.header("Current Model Metrics")
    st.sidebar.write(f"Training Accuracy: {train_data_acc:.2f}")
    st.sidebar.write(f"Testing Accuracy: {test_data_acc:.2f}")
    st.sidebar.write(f"Precision: {test_precision:.2f}")
    st.sidebar.write(f"Recall: {test_recall:.2f}")
    st.sidebar.write(f"F1 Score: {test_f1:.2f}")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file for prediction", type="csv")

if uploaded_file is not None:
    try:
        input_data = pd.read_csv(uploaded_file)
        if 'Class' in input_data.columns:
            input_data = input_data.drop('Class', axis=1)
        if input_data.shape[1] == X.shape[1]:
            predictions = model.predict(input_data)
            st.write(predictions)
            st.success("Predictions made successfully")
            for i, pred in enumerate(predictions):
                log_prediction(input_data.iloc[i].values.tolist(), pred)
        else:
            st.error(f"Expected {X.shape[1]} features, but got {input_data.shape[1]}")
    except Exception as e:
        st.error(f"Error reading the file: {e}")
else:
    # Create columns for input layout
    cols = st.columns((2, 1))

    with cols[0]:
        input_data = st.text_area(
            "Enter the transaction features separated by commas", 
            st.session_state.input_data,
            help="Provide the feature values as a comma-separated list (e.g., 0.1, 0.2, ..., 0.3)"
        )
    with cols[1]:
        st.write("Feature Values (0 to 100)")
        feature_slider = st.slider('Adjust feature values here', 0, 100, (0, 100))

    submit = st.button("Submit")
    clear_input = st.button("Clear Input")

    if clear_input:
        st.session_state.input_data = ""
        st.experimental_rerun()

    if submit:
        with st.spinner('Making prediction...'):
            try:
                # Get input feature values
                features = np.array([float(x) for x in input_data.split(',')], dtype=np.float64)

                # Check if the input features length matches the model input
                if len(features) != X.shape[1]:
                    st.error(f"Expected {X.shape[1]} features, but got {len(features)}")
                else:
                    # Make prediction
                    prediction = model.predict(features.reshape(1, -1))

                    # Log the prediction
                    log_prediction(features.tolist(), prediction[0])

                    # Display result
                    if prediction[0] == 0:
                        st.success("Legitimate transaction")
                    else:
                        st.error("Fraudulent transaction")
            except ValueError:
                st.error("Invalid input. Please enter numeric values separated by commas.")

# Visualize input data
if st.session_state.input_data:
    st.subheader("Input Data Visualization")
    input_values = np.array([float(x) for x in st.session_state.input_data.split(',')], dtype=np.float64)
    input_df = pd.DataFrame(input_values.reshape(1, -1), columns=X.columns)
    st.bar_chart(input_df.T)

# Footer
st.markdown("""
---
Note: This app is for demonstration purposes only. Always consult with a financial expert for fraud detection in real-world scenarios.
""")
