# Mall Customer Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange.svg)](https://scikit-learn.org/)

This project performs customer segmentation on mall customer data using K-Means clustering. It includes both a command-line analysis pipeline and an interactive web dashboard built with Flask.

## 🚀 Features

- **Data Exploration & Preprocessing**
- **Automatic Optimal Cluster Detection** (Elbow Method & Silhouette Score)
- **2D & 3D Visualizations** of customer segments
- **Detailed Cluster Analysis** with demographic and behavioral insights
- **Marketing Strategy Recommendations** based on segment characteristics
- **Interactive Web Dashboard** for exploring segmentation results

## 📋 Dataset

The analysis uses the Mall_Customers.csv dataset with the following features:
- CustomerID: Unique identifier for each customer
- Gender: Customer's gender (Male/Female)
- Age: Customer's age
- Annual Income (k$): Customer's annual income in thousands of dollars
- Spending Score (1-100): Score assigned by the mall based on customer behavior and spending patterns

## 🔍 Installation

1. Clone the repository:
   ```
   git clone https://github.com/username/mall-customer-segmentation.git
   cd mall-customer-segmentation
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## 🖥️ Usage

### Command-line Analysis

To run the complete analysis pipeline from the command line:
```
python main.py
```

This will generate visualizations and CSV files with the segmentation results.

### Web Dashboard

To launch the interactive web dashboard:
```
python run_flask_app.py
```

This will start a Flask web server and open a browser tab at http://127.0.0.1:5000/ with the application.

## 📁 Project Structure

```
mall-customer-segmentation/
│
├── Mall_Customers.csv       # Input dataset
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
│
├── data_preprocessing.py    # Data loading and preprocessing
├── kmeans_clustering.py     # K-means clustering implementation
├── advanced_analysis.py     # Multi-dimensional clustering
├── main.py                  # CLI analysis pipeline
│
├── app.py                   # Flask web application
├── run_flask_app.py         # Script to run the web app
│
└── templates/               # HTML templates for Flask
    ├── index.html           # Landing page
    ├── results.html         # Analysis results page
    └── error.html           # Error page
```

## 📈 Results

The K-means clustering algorithm identifies distinct customer segments based on annual income and spending patterns. For each segment, the analysis provides:

1. **Demographic Profile**: Age distribution, gender ratio, income levels
2. **Spending Behavior**: Average spending score and patterns
3. **Marketing Recommendations**: Tailored strategies for each segment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 References

- [scikit-learn K-means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Matplotlib Visualization](https://matplotlib.org/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📝 Notes for GitHub Publication

Before pushing to GitHub:

1. **Create a .gitignore file** with the following content:
   ```
   # Python bytecode
   __pycache__/
   *.py[cod]
   *$py.class
   
   # Virtual environments
   venv/
   env/
   ENV/
   
   # Generated files
   *.png
   *.csv
   !Mall_Customers.csv
   
   # Flask session files
   flask_session/
   
   # IDE files
   .idea/
   .vscode/
   *.swp
   *.swo
   ```

2. **Create a screenshots directory** for README images
   ```
   mkdir screenshots
   ```

3. **Run the analysis once** to generate screenshots for documentation:
   ```
   python main.py
   ```

4. **Copy key visualizations** to the screenshots directory for README use
   ```
   copy customer_segments_5_clusters.png screenshots/customer_segments.png
   ```

5. **Update repository URL** in this README with your actual GitHub username/repository 
