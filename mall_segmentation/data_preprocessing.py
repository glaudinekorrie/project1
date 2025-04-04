import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load the mall customer data from CSV file"""
    print("Loading data from:", file_path)
    df = pd.read_csv(file_path)
    print("Data loaded successfully. Shape:", df.shape)
    return df

def explore_data(df):
    """Explore the dataset and print basic statistics"""
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    print("\nData Information:")
    print(df.info())
    
    print("\nStatistical Summary:")
    print(df.describe())
    
    print("\nChecking for missing values:")
    print(df.isnull().sum())
    
    return df

def preprocess_data(df):
    """Preprocess the data for clustering"""
    # For this clustering problem, we'll focus on Annual Income and Spending Score
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    
    # We could also standardize the data, but in this case it's not necessary
    # since both features are already in similar scales
    
    return X

if __name__ == "__main__":
    # Load the dataset
    df = load_data("Mall_Customers.csv")
    
    # Explore the data
    df = explore_data(df)
    
    # Preprocess the data for clustering
    X = preprocess_data(df)
    
    print("Data preprocessing completed successfully.")
    print("Preprocessed data shape:", X.shape) 