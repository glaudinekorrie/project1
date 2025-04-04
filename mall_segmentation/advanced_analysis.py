import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data_preprocessing import load_data

def perform_multidimensional_clustering(df, n_clusters=5):
    """
    Perform clustering using multiple dimensions (Age, Annual Income, Spending Score)
    """
    # Select the features for clustering
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    
    # Standardize the data to have mean=0 and variance=1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', 
                   max_iter=300, n_init=10, random_state=42)
    df['Cluster_Multi'] = kmeans.fit_predict(X_scaled)
    
    return df, kmeans.cluster_centers_

def create_pairplot(df):
    """
    Create a pairplot to visualize relationships between features colored by cluster
    """
    # Create a subset of the dataframe with numeric columns and cluster
    plot_df = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster_Multi']]
    
    # Create the pairplot
    plt.figure(figsize=(12, 10))
    sns.pairplot(plot_df, hue='Cluster_Multi', palette='tab10')
    plt.savefig('cluster_pairplot.png')
    plt.close()

def create_3d_plot(df):
    """
    Create a 3D scatter plot to visualize the clusters in 3D space
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the clusters
    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange', 'brown']
    
    for cluster_id in range(df['Cluster_Multi'].nunique()):
        cluster_data = df[df['Cluster_Multi'] == cluster_id]
        ax.scatter(
            cluster_data['Age'], 
            cluster_data['Annual Income (k$)'], 
            cluster_data['Spending Score (1-100)'],
            s=80, c=colors[cluster_id], label=f'Cluster {cluster_id+1}'
        )
    
    # Set labels and title
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income (k$)')
    ax.set_zlabel('Spending Score (1-100)')
    ax.set_title('3D Visualization of Customer Segments')
    
    plt.legend()
    plt.savefig('3d_clusters.png')
    plt.close()

def analyze_gender_distribution(df):
    """
    Analyze the gender distribution in each cluster
    """
    # Count gender in each cluster
    gender_cluster = pd.crosstab(df['Cluster_Multi'], df['Gender'])
    
    # Calculate the percentage
    gender_cluster_percent = gender_cluster.div(gender_cluster.sum(axis=1), axis=0) * 100
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    gender_cluster_percent.plot(kind='bar', stacked=True, colormap='Blues')
    plt.title('Gender Distribution in Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('gender_distribution.png')
    plt.close()
    
    return gender_cluster, gender_cluster_percent

def generate_cluster_profiles(df):
    """
    Generate detailed profiles for each cluster
    """
    profiles = []
    
    for cluster_id in range(df['Cluster_Multi'].nunique()):
        cluster_data = df[df['Cluster_Multi'] == cluster_id]
        
        # Calculate statistics
        profile = {
            'Cluster': f'Cluster {cluster_id+1}',
            'Size': len(cluster_data),
            'Percentage': f"{(len(cluster_data) / len(df) * 100):.2f}%",
            'Age (Mean)': f"{cluster_data['Age'].mean():.2f}",
            'Age (Min-Max)': f"{cluster_data['Age'].min()}-{cluster_data['Age'].max()}",
            'Income (Mean)': f"{cluster_data['Annual Income (k$)'].mean():.2f}",
            'Income (Min-Max)': f"{cluster_data['Annual Income (k$)'].min()}-{cluster_data['Annual Income (k$)'].max()}",
            'Spending (Mean)': f"{cluster_data['Spending Score (1-100)'].mean():.2f}",
            'Spending (Min-Max)': f"{cluster_data['Spending Score (1-100)'].min()}-{cluster_data['Spending Score (1-100)'].max()}",
            'Males (%)': f"{(len(cluster_data[cluster_data['Gender'] == 'Male']) / len(cluster_data) * 100):.2f}%",
            'Females (%)': f"{(len(cluster_data[cluster_data['Gender'] == 'Female']) / len(cluster_data) * 100):.2f}%"
        }
        
        profiles.append(profile)
    
    # Create a dataframe for the profiles
    profile_df = pd.DataFrame(profiles)
    
    # Save the profiles to CSV
    profile_df.to_csv('cluster_profiles.csv', index=False)
    
    return profile_df

def suggest_marketing_strategies(profiles):
    """
    Suggest marketing strategies based on cluster profiles
    """
    strategies = []
    
    for _, profile in profiles.iterrows():
        cluster = profile['Cluster']
        
        # Convert string values to numeric for analysis
        age_mean = float(profile['Age (Mean)'])
        income_mean = float(profile['Income (Mean)'])
        spending_mean = float(profile['Spending (Mean)'])
        
        strategy = {'Cluster': cluster, 'Characteristics': '', 'Suggested Strategies': ''}
        
        # Define characteristics based on the profile
        characteristics = []
        marketing_strategies = []
        
        # Age-based characteristics
        if age_mean < 30:
            characteristics.append("Young customers")
            marketing_strategies.append("Use social media marketing and mobile apps")
        elif age_mean < 45:
            characteristics.append("Middle-aged customers")
            marketing_strategies.append("Balance between digital and traditional marketing")
        else:
            characteristics.append("Older customers")
            marketing_strategies.append("Focus on loyalty programs and personalized services")
        
        # Income-based characteristics
        if income_mean < 40:
            characteristics.append("Lower income")
            marketing_strategies.append("Offer budget-friendly options and promotions")
        elif income_mean < 70:
            characteristics.append("Average income")
            marketing_strategies.append("Provide value-based offerings with occasional premium options")
        else:
            characteristics.append("Higher income")
            marketing_strategies.append("Highlight premium products and exclusive experiences")
        
        # Spending-based characteristics
        if spending_mean < 40:
            characteristics.append("Low spending")
            marketing_strategies.append("Implement periodic discounts and incentives to increase spending")
        elif spending_mean < 70:
            characteristics.append("Moderate spending")
            marketing_strategies.append("Upselling and cross-selling strategies")
        else:
            characteristics.append("High spending")
            marketing_strategies.append("Premium customer experience and early access to new products")
        
        strategy['Characteristics'] = ", ".join(characteristics)
        strategy['Suggested Strategies'] = "; ".join(marketing_strategies)
        
        strategies.append(strategy)
    
    # Create a dataframe for the strategies
    strategy_df = pd.DataFrame(strategies)
    
    # Save the strategies to CSV
    strategy_df.to_csv('marketing_strategies.csv', index=False)
    
    return strategy_df

if __name__ == "__main__":
    # Load the data
    df = load_data("Mall_Customers.csv")
    
    # Perform multidimensional clustering
    df_clustered, centroids = perform_multidimensional_clustering(df)
    
    # Create visualizations
    create_pairplot(df_clustered)
    create_3d_plot(df_clustered)
    
    # Analyze gender distribution
    gender_count, gender_percent = analyze_gender_distribution(df_clustered)
    print("\nGender Distribution by Cluster (%):")
    print(gender_percent)
    
    # Generate cluster profiles
    profiles = generate_cluster_profiles(df_clustered)
    print("\nCluster Profiles:")
    print(profiles)
    
    # Suggest marketing strategies
    strategies = suggest_marketing_strategies(profiles)
    print("\nSuggested Marketing Strategies:")
    print(strategies) 