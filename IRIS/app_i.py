import pandas as pd
import numpy as np
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load and prepare dataset
data = pd.read_csv('IRIS.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Standardize and train model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# Streamlit UI
st.title("🌸 Iris Flower Species Predictor")

# User inputs
sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.0)
sepal_width  = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
petal_width  = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

if st.button("Predict"):
    user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    user_input_scaled = scaler.transform(user_input)
    prediction = knn.predict(user_input_scaled)
    st.success(f"The predicted species is: **{prediction[0]}**")
