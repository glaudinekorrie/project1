import os
import pandas as pd
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import seaborn as sns
from flask import Flask, render_template, request, jsonify, send_file

from data_preprocessing import load_data, preprocess_data
from kmeans_clustering import find_optimal_clusters, perform_kmeans_clustering, visualize_clusters, analyze_clusters
from advanced_analysis import (
    perform_multidimensional_clustering,
    create_3d_plot,
    analyze_gender_distribution,
    generate_cluster_profiles,
    suggest_marketing_strategies
)

app = Flask(__name__, template_folder='templates')

# Global variables to store data and results
data = None
analysis_results = {}

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return base64.b64encode(image_png).decode('utf-8')

def create_visualization(X, y_kmeans, centroids, n_clusters):
    """Create customer segments visualization and return as base64 image"""
    fig = plt.figure(figsize=(10, 8))
    
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
    
    img_str = fig_to_base64(fig)
    plt.close(fig)
    
    return img_str

def create_elbow_plot(wcss, silhouette_scores):
    """Create elbow method plot and return as base64 image"""
    fig = plt.figure(figsize=(12, 5))
    
    # Plot 1: WCSS (Elbow Method)
    plt.subplot(1, 2, 1)
    plt.plot(range(2, len(wcss) + 2), wcss, marker='o', linestyle='-')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    
    # Plot 2: Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o', linestyle='-')
    plt.title('Silhouette Score Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    
    plt.tight_layout()
    
    img_str = fig_to_base64(fig)
    plt.close(fig)
    
    return img_str

def create_gender_distribution_plot(df):
    """Create gender distribution plot and return as base64 image"""
    try:
        # Count gender in each cluster
        gender_cluster = pd.crosstab(df['Cluster_Multi'], df['Gender'])
        
        # Calculate the percentage
        gender_cluster_percent = gender_cluster.div(gender_cluster.sum(axis=1), axis=0) * 100
        
        # Create a new figure
        plt.figure(figsize=(10, 6))
        
        # Create the stacked bar chart using pandas plot
        ax = gender_cluster_percent.plot(kind='bar', stacked=True, colormap='Blues')
        
        # Add labels and title
        plt.title('Gender Distribution in Clusters')
        plt.xlabel('Cluster')
        plt.ylabel('Percentage (%)')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Gender')
        
        # Ensure figure is rendered completely
        plt.tight_layout()
        
        # Get the current figure
        fig = plt.gcf()
        
        # Convert to base64
        img_str = fig_to_base64(fig)
        plt.close(fig)
        
        return img_str
    except Exception as e:
        print(f"Error creating gender distribution plot: {e}")
        # Create a simple error figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error creating gender plot: {e}", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        img_str = fig_to_base64(fig)
        plt.close(fig)
        return img_str

def create_3d_visualization(df):
    """Create 3D visualization and return as base64 image"""
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
    
    img_str = fig_to_base64(fig)
    plt.close(fig)
    
    return img_str

def run_analysis():
    """Run the customer segmentation analysis and store results"""
    global data, analysis_results
    
    try:
        # Load and preprocess data
        df = load_data("Mall_Customers.csv")
        X = preprocess_data(df)
        
        # Find optimal number of clusters
        optimal_clusters, wcss, silhouette_scores = find_optimal_clusters(X)
        
        # Perform K-means clustering
        y_kmeans, centroids = perform_kmeans_clustering(X, optimal_clusters)
        
        # Analyze clusters
        analysis_df = analyze_clusters(df, y_kmeans, optimal_clusters)
        
        # Perform multidimensional clustering
        df_clustered, _ = perform_multidimensional_clustering(df, n_clusters=5)
        
        # Generate cluster profiles
        profiles = generate_cluster_profiles(df_clustered)
        
        # Generate marketing strategies
        strategies = suggest_marketing_strategies(profiles)
        
        # Store results
        data = df
        analysis_results = {
            'optimal_clusters': optimal_clusters,
            'wcss': wcss,
            'silhouette_scores': silhouette_scores,
            'y_kmeans': y_kmeans.tolist(),
            'centroids': centroids.tolist(),
            'elbow_plot': create_elbow_plot(wcss, silhouette_scores),
            'cluster_plot': create_visualization(X, y_kmeans, centroids, optimal_clusters),
            'analysis': analysis_df.to_dict('records'),
            'df_clustered': df_clustered,
            '3d_plot': create_3d_visualization(df_clustered),
            'gender_plot': create_gender_distribution_plot(df_clustered),
            'profiles': profiles.to_dict('records'),
            'strategies': strategies.to_dict('records')
        }
        
        return True
    except Exception as e:
        print(f"Error running analysis: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run-analysis', methods=['POST'])
def api_run_analysis():
    success = run_analysis()
    if success:
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'message': 'Analysis failed'})

@app.route('/results')
def results():
    if data is None or not analysis_results:
        return render_template('error.html', message="No analysis has been run yet")
    
    return render_template('results.html', 
                          optimal_clusters=analysis_results['optimal_clusters'],
                          elbow_plot=analysis_results['elbow_plot'],
                          cluster_plot=analysis_results['cluster_plot'],
                          analysis=analysis_results['analysis'],
                          plot_3d=analysis_results['3d_plot'],
                          gender_plot=analysis_results['gender_plot'],
                          profiles=analysis_results['profiles'],
                          strategies=analysis_results['strategies'])

@app.route('/data')
def get_data():
    try:
        # Load data directly from the file, regardless of whether the analysis has been run
        # This ensures the data route works independently
        data_df = pd.read_csv("Mall_Customers.csv")
        return render_template('data.html', data=data_df.to_dict('records'))
    except Exception as e:
        return jsonify({'error': f'Error loading data: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True) 