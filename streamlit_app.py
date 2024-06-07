import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
@st.cache
def load_data():
    data = pd.read_csv('airline_passenger_satisfaction.csv')
    return data

# Exploratory Data Analysis
def eda(data):
    st.subheader("Exploratory Data Analysis")
    st.write("## Data Overview")
    st.write(data.head())
    
    st.write("## Summary Statistics")
    st.write(data.describe())
    
    st.write("## Data Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='satisfaction', data=data, ax=ax)
    st.pyplot(fig)
    
    st.write("## Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Preprocessing
def preprocess_data(data):
    # Drop missing values
    data = data.dropna()
    
    # Convert categorical variables to dummy/indicator variables
    data = pd.get_dummies(data, drop_first=True)
    
    return data

# Machine Learning Model
def build_model(data):
    st.subheader("Machine Learning Model")
    
    # Splitting the data
    X = data.drop('satisfaction_satisfied', axis=1)
    y = data['satisfaction_satisfied']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Model performance
    st.write("### Accuracy Score")
    st.write(accuracy_score(y_test, y_pred))
    
    st.write("### Classification Report")
    st.write(classification_report(y_test, y_pred))
    
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Main function to run the app
def main():
    st.title("Airline Customer Satisfaction Analysis")
    
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Choose an option", ['Introduction', 'EDA', 'Model'])
    
    data = load_data()
    
    if option == 'Introduction':
        st.write("""
        # Airline Customer Satisfaction Analysis
        This application provides an analysis of airline customer satisfaction. You can explore the dataset, 
        perform exploratory data analysis (EDA), and build a machine learning model to predict customer satisfaction.
        """)
    
    elif option == 'EDA':
        eda(data)
    
    elif option == 'Model':
        data = preprocess_data(data)
        build_model(data)

if __name__ == '__main__':
    main()


