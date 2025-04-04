import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from data_preprocessing import load_data, preprocess_data

def find_optimal_clusters(X, max_clusters=10):
    """
    Find the optimal number of clusters using the Elbow method
    and Silhouette score
    """
    wcss = []  # Within-Cluster Sum of Square
    silhouette_scores = []
    
    # Try different numbers of clusters from 2 to max_clusters
    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', 
                        max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        
        # Calculate silhouette score
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(X, labels))
    
    # Plot the Elbow method graph
    plt.figure(figsize=(12, 5))
    
    # Plot 1: WCSS (Elbow Method)
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), wcss, marker='o', linestyle='-')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    
    # Plot 2: Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', linestyle='-')
    plt.title('Silhouette Score Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimal_clusters.png')
    plt.close()
    
    # Return the optimal number of clusters (based on silhouette score)
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    
    return optimal_clusters, wcss, silhouette_scores

def perform_kmeans_clustering(X, n_clusters):
    """
    Perform K-means clustering with the given number of clusters
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', 
                    max_iter=300, n_init=10, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    
    # Get the centroids
    centroids = kmeans.cluster_centers_
    
    return y_kmeans, centroids

def visualize_clusters(X, y_kmeans, centroids, n_clusters):
    """
    Visualize the clusters and their centroids
    """
    plt.figure(figsize=(10, 8))
    
    # Plot the clusters with different colors
    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange', 'brown']
    
    for i in range(n_clusters):
        plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], 
                   s=100, c=colors[i], label=f'Cluster {i+1}')
    
    # Plot the centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], 
               s=300, c='grey', marker='*', label='Centroids')
    
    plt.title(f'Customer Segments (K={n_clusters})')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'customer_segments_{n_clusters}_clusters.png')
    plt.close()

def analyze_clusters(df, y_kmeans, n_clusters):
    """
    Analyze the characteristics of each cluster
    """
    # Add cluster labels to the dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = y_kmeans
    
    # Analyze each cluster
    cluster_analysis = []
    
    for i in range(n_clusters):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == i]
        
        analysis = {
            'Cluster': f'Cluster {i+1}',
            'Size': len(cluster_data),
            'Percentage': f"{(len(cluster_data) / len(df) * 100):.2f}%",
            'Avg Age': f"{cluster_data['Age'].mean():.2f}",
            'Avg Annual Income': f"{cluster_data['Annual Income (k$)'].mean():.2f}",
            'Avg Spending Score': f"{cluster_data['Spending Score (1-100)'].mean():.2f}",
            'Males': f"{len(cluster_data[cluster_data['Gender'] == 'Male'])}/{len(cluster_data)}",
            'Females': f"{len(cluster_data[cluster_data['Gender'] == 'Female'])}/{len(cluster_data)}"
        }
        
        cluster_analysis.append(analysis)
    
    # Create a dataframe for the analysis
    analysis_df = pd.DataFrame(cluster_analysis)
    
    # Save the analysis to CSV
    analysis_df.to_csv('cluster_analysis.csv', index=False)
    
    return analysis_df

if __name__ == "__main__":
    # Load and preprocess the data
    df = load_data("Mall_Customers.csv")
    X = preprocess_data(df)
    
    # Find the optimal number of clusters
    optimal_clusters, wcss, silhouette_scores = find_optimal_clusters(X)
    print(f"Optimal number of clusters: {optimal_clusters}")
    
    # Perform K-means clustering with the optimal number of clusters
    y_kmeans, centroids = perform_kmeans_clustering(X, optimal_clusters)
    
    # Visualize the clusters
    visualize_clusters(X, y_kmeans, centroids, optimal_clusters)
    
    # Analyze the clusters
    analysis_df = analyze_clusters(df, y_kmeans, optimal_clusters)
    print("\nCluster Analysis:")
    print(analysis_df) 