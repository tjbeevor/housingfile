import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('California Housing Dataset Explorer')

# Load the dataset
@st.cache_data  # This will cache the data loading
def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df

try:
    # Load the data
    df = load_data()
    
    st.write("## Dataset Overview")
    st.write(f"Shape of dataset: {df.shape}")
    
    # Show first few rows
    st.write("### First few rows of the dataset")
    st.dataframe(df.head())
    
    # Basic statistics
    st.write("### Basic Statistics")
    st.dataframe(df.describe())
    
    # Correlation heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Distribution plots
    st.write("### Feature Distributions")
    feature = st.selectbox("Select feature to view distribution:", df.columns)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x=feature, bins=50, ax=ax)
    plt.title(f'Distribution of {feature}')
    st.pyplot(fig)
    
    # Scatter plot
    st.write("### Scatter Plot")
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Select X-axis feature:", df.columns)
    with col2:
        y_axis = st.selectbox("Select Y-axis feature:", df.columns, index=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_axis, y=y_axis, alpha=0.5)
    plt.title(f'{y_axis} vs {x_axis}')
    st.pyplot(fig)
    
    # Save option
    if st.button('Download Dataset as CSV'):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Click to Download",
            data=csv,
            file_name='california_housing.csv',
            mime='text/csv',
        )

except Exception as e:
    st.error(f"""
    An error occurred while loading the dataset. 
    Please make sure you have scikit-learn installed:
    ```
    pip install scikit-learn
    ```
    Error details: {str(e)}
    """)

st.write("""
### Dataset Description
This dataset contains information about housing in California:
- **MedInc**: Median income in block group
- **HouseAge**: Median house age in block group
- **AveRooms**: Average number of rooms per household
- **AveBedrms**: Average number of bedrooms per household
- **Population**: Block group population
- **AveOccup**: Average number of household members
- **Latitude**: Block group latitude
- **Longitude**: Block group longitude
- **MedHouseVal**: Median house value (target variable)
""")
