import os
import time
import pandas as pd
from data_preprocessing import load_data, explore_data, preprocess_data
from kmeans_clustering import find_optimal_clusters, perform_kmeans_clustering, visualize_clusters, analyze_clusters
from advanced_analysis import (
    perform_multidimensional_clustering, 
    create_pairplot, 
    create_3d_plot, 
    analyze_gender_distribution,
    generate_cluster_profiles,
    suggest_marketing_strategies
)

def create_output_dir():
    """Create output directory for results"""
    if not os.path.exists('results'):
        os.makedirs('results')
        print("Created 'results' directory")


def run_segmentation_pipeline():
    """Run the complete customer segmentation pipeline"""
    start_time = time.time()
    
    print("="*80)
    print("CUSTOMER SEGMENTATION PROJECT")
    print("="*80)
    
    # Create output directory
    create_output_dir()
    
    # Step 1: Load and explore data
    print("\n[Step 1] Loading and exploring data...")
    df = load_data("Mall_Customers.csv")
    df = explore_data(df)
    
    # Step 2: Basic preprocessing
    print("\n[Step 2] Preprocessing data...")
    X = preprocess_data(df)
    
    # Step 3: Find optimal number of clusters
    print("\n[Step 3] Finding optimal number of clusters...")
    optimal_clusters, wcss, silhouette_scores = find_optimal_clusters(X)
    print(f"Optimal number of clusters based on silhouette score: {optimal_clusters}")
    
    # Step 4: Perform K-means clustering with the optimal number of clusters
    print(f"\n[Step 4] Performing K-means clustering with {optimal_clusters} clusters...")
    y_kmeans, centroids = perform_kmeans_clustering(X, optimal_clusters)
    
    # Step 5: Visualize the clusters
    print("\n[Step 5] Visualizing clusters...")
    visualize_clusters(X, y_kmeans, centroids, optimal_clusters)
    print(f"Saved visualization to 'customer_segments_{optimal_clusters}_clusters.png'")
    
    # Step 6: Analyze the clusters
    print("\n[Step 6] Analyzing clusters...")
    analysis_df = analyze_clusters(df, y_kmeans, optimal_clusters)
    print("Cluster analysis results:")
    print(analysis_df)
    print("Saved analysis to 'cluster_analysis.csv'")
    
    # Step 7: Advanced multi-dimensional clustering
    print("\n[Step 7] Performing multi-dimensional clustering...")
    df_clustered, multi_centroids = perform_multidimensional_clustering(df, n_clusters=5)
    
    # Step 8: Advanced visualizations
    print("\n[Step 8] Creating advanced visualizations...")
    create_pairplot(df_clustered)
    print("Saved pairplot to 'cluster_pairplot.png'")
    
    create_3d_plot(df_clustered)
    print("Saved 3D plot to '3d_clusters.png'")
    
    # Step 9: Gender analysis
    print("\n[Step 9] Analyzing gender distribution...")
    gender_count, gender_percent = analyze_gender_distribution(df_clustered)
    print("Gender distribution by cluster (%):")
    print(gender_percent)
    print("Saved gender distribution visualization to 'gender_distribution.png'")
    
    # Step 10: Detailed cluster profiles
    print("\n[Step 10] Generating detailed cluster profiles...")
    profiles = generate_cluster_profiles(df_clustered)
    print("Cluster profiles:")
    print(profiles)
    print("Saved profiles to 'cluster_profiles.csv'")
    
    # Step 11: Marketing strategy suggestions
    print("\n[Step 11] Suggesting marketing strategies...")
    strategies = suggest_marketing_strategies(profiles)
    print("Marketing strategy suggestions:")
    print(strategies)
    print("Saved marketing strategies to 'marketing_strategies.csv'")
    
    # Print execution time
    execution_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"Pipeline completed in {execution_time:.2f} seconds")
    print("="*80)
    
    return df_clustered

if __name__ == "__main__":
    segmented_data = run_segmentation_pipeline()
    print("\nCustomer segmentation completed successfully!") 