import os
import sys
import webbrowser
from threading import Timer

def open_browser():
    """Open browser tab with the application"""
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == "__main__":
    # Make sure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Copy data file if needed
    if not os.path.exists('Mall_Customers.csv'):
        parent_dir = os.path.dirname(script_dir)
        if os.path.exists(os.path.join(parent_dir, 'Mall_Customers.csv')):
            import shutil
            shutil.copy(os.path.join(parent_dir, 'Mall_Customers.csv'), 'Mall_Customers.csv')
            print("Copied Mall_Customers.csv to working directory")
    
    # Open browser tab after a delay
    Timer(1.5, open_browser).start()
    
    # Run the Flask app
    print("Starting Mall Customer Segmentation Web Application...")
    print("Navigate to http://127.0.0.1:5000/ if the browser doesn't open automatically")
    from app import app
    app.run(debug=True) 