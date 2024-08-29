import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load your trained model
model = load_model(import streamlit as st)
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load your trained model
model = load_model('demand prediction.keras')

# Initialize the scaler (replace this with your actual scaler if needed)
scaler = StandardScaler()

# Streamlit app title
st.title('Demand Prediction')

# Sidebar for input features
st.sidebar.header('Input Features')
year = st.sidebar.number_input('Year', min_value=2000, max_value=2030, value=2024)
location = st.sidebar.selectbox('Location', options=[1, 2, 3])  # Adjust options based on your dataset
week = st.sidebar.number_input('Week', min_value=1, max_value=52, value=1)
time = st.sidebar.number_input('Time', min_value=0, max_value=24, value=12)  # Adjust based on your time data

# Create a DataFrame for input
input_data = pd.DataFrame([[year, location, week, time]], columns=['year', 'location', 'week', 'time'])

# Preprocess the input data (scaling)
input_scaled = scaler.fit_transform(input_data)  # You may want to load your scaler used during training instead
input_scaled = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))  # Reshape for LSTM input

# Predict using the model
if st.button('Predict Demand'):
    prediction = model.predict(input_scaled)
    st.write(f'Predicted Demand: {prediction[0][0]:.2f}')

# Add an optional section to show instructions or note

# Initialize the scaler (replace this with your actual scaler if needed)
scaler = StandardScaler()

# Streamlit app title
st.title('Demand Prediction')

# Sidebar for input features
st.sidebar.header('Input Features')
year = st.sidebar.number_input('Year', min_value=2000, max_value=2030, value=2024)
location = st.sidebar.selectbox('Location', options=[1, 2, 3])  # Adjust options based on your dataset
week = st.sidebar.number_input('Week', min_value=1, max_value=52, value=1)
time = st.sidebar.number_input('Time', min_value=0, max_value=24, value=12)  # Adjust based on your time data

# Create a DataFrame for input
input_data = pd.DataFrame([[year, location, week, time]], columns=['year', 'location', 'week', 'time'])

# Preprocess the input data (scaling)
input_scaled = scaler.fit_transform(input_data)  # You may want to load your scaler used during training instead
input_scaled = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))  # Reshape for LSTM input

# Predict using the model
if st.button('Predict Demand'):
    prediction = model.predict(input_scaled)
    st.write(f'Predicted Demand: {prediction[0][0]:.2f}')

# Add an optional section to show instructions or notes
st.markdown("""
### Notes:
- Ensure you input reasonable values for each feature.
- The model predicts based on the features provided; adjustments may be required for different feature ranges.
""")
