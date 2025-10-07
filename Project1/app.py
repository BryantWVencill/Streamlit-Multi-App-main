import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Loading and Caching ---
@st.cache_data
def load_data():
    """Loads the Iris dataset."""
    iris = load_iris()
    return iris.data, iris.target, iris.target_names, iris.feature_names

@st.cache_resource
def train_model(X_train, y_train, hidden_layer_sizes, activation, max_iter, random_state):
    """Trains the MLPClassifier and returns the trained model and scaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        max_iter=max_iter,
        random_state=random_state
    )
    mlp.fit(X_train_scaled, y_train)
    return mlp, scaler

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Iris ANN Classifier", layout="wide")
    
    # --- Title and Introduction ---
    st.title("🌸 Interactive Artificial Neural Network Classifier")
    st.write(
        "This app allows you to train and evaluate a Multi-Layer Perceptron (MLP) classifier "
        "on the classic Iris dataset. Adjust the hyperparameters in the sidebar to see how they "
        "affect the model's performance and make your own predictions!"
    )

    # --- Load Data ---
    X, y, target_names, feature_names = load_data()

    # --- Sidebar for Hyperparameters ---
    st.sidebar.header("🔧 Model Hyperparameters")
    
    hidden_layer_sizes_str = st.sidebar.text_input(
        "Hidden Layer Sizes (comma-separated)", "10,5",
        help="Enter the number of neurons for each hidden layer, separated by commas. E.g., '100' for one layer with 100 neurons, or '50,25' for two layers."
    )
    try:
        hidden_layer_sizes = tuple(int(x.strip()) for x in hidden_layer_sizes_str.split(','))
    except ValueError:
        st.sidebar.error("Please enter a valid comma-separated list of integers.")
        return

    activation = st.sidebar.selectbox(
        "Activation Function",
        ('relu', 'identity', 'logistic', 'tanh'),
        index=0
    )
    max_iter = st.sidebar.slider(
        "Maximum Iterations", 200, 2000, 500, 100
    )
    model_random_state = st.sidebar.number_input(
        "Model Random State", value=42, step=1
    )
    
    st.sidebar.header("📊 Data Splitting")
    split_random_state = st.sidebar.number_input(
        "Train-Test Split Random State", value=42, step=1
    )

    # --- Data Splitting ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=split_random_state)
    
    # --- Model Training ---
    with st.spinner('Training the model... Please wait.'):
        mlp, scaler = train_model(X_train, y_train, hidden_layer_sizes, activation, max_iter, model_random_state)

    st.success("Model trained successfully!")

    # --- Model Evaluation ---
    st.header("📊 Model Evaluation")
    
    X_test_scaled = scaler.transform(X_test)
    predictions = mlp.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Accuracy")
        st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")
        st.write("Accuracy is the proportion of correctly classified samples.")

    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names, ax=ax)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        st.pyplot(fig)

    # --- Interactive Prediction ---
    st.sidebar.header("🌱 Make a Prediction")
    st.sidebar.write("Use the sliders to select the features of an Iris flower.")

    sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
    sepal_width = st.sidebar.slider("Sepal Width (cm)", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
    petal_length = st.sidebar.slider("Petal Length (cm)", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
    petal_width = st.sidebar.slider("Petal Width (cm)", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

    user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    user_input_scaled = scaler.transform(user_input)
    
    prediction = mlp.predict(user_input_scaled)
    prediction_proba = mlp.predict_proba(user_input_scaled)

    st.header("🔮 Prediction Results")
    st.write("Based on the input features, the model predicts the following Iris species:")

    predicted_species = target_names[prediction[0]]
    st.markdown(f"## **{predicted_species.capitalize()}**")

    st.write("### Prediction Probabilities")
    proba_df = pd.DataFrame(prediction_proba, columns=target_names)
    st.dataframe(proba_df.style.format("{:.2%}"))

if __name__ == "__main__":
    main()
