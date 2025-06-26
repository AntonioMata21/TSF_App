# Time Series Forecaster 📈
A web application built with Streamlit that enables the application of various forecasting methods and models for time series data through an interactive and visual interface.
Description
This application integrates multiple forecasting techniques, from traditional statistical methods to machine learning and deep learning algorithms, providing an intuitive interface for analyzing and predicting time series data.

## Implemented Methods

Statistical Models: ARIMA, Exponential Smoothing, Simple Moving Average (SMA)
Machine Learning: LightGBM, XGBoost, Random Forest
Deep Learning: Recurrent Neural Networks (RNN), LSTM
Statistical Analysis: Stationarity tests, ACF/PACF plots
Evaluation Metrics: MAE, MSE, MAPE

# Features
✅ Interactive and user-friendly web interface
✅ Custom CSV data upload
✅ Automatic time series visualizations
✅ Multiple model comparison
✅ Export results and charts
✅ Detailed evaluation metrics

# Installation
# rerequisites

Python 3.7 or higher
pip (Python package manager)

# Step 1: Clone or download the repository
bashgit clone https://github.com/AntonioMata21/TSF_App.git
cd time-series-forecaster
# Step 2: Install dependencies
# Quick installation with requirements.txt
bashpip install -r requirements.txt

# Usage
# Run the application
bashstreamlit run app_tsf.py
Note: If the streamlit command is not recognized, use:
bashpython -m streamlit run app_tsf.py
# Access the application
Once the command is executed, it will automatically open in your web browser. If it doesn't open automatically, navigate to:
http://localhost:8501
# Steps to use the application

Load data: Upload your CSV file with the time series data
Configure parameters: Select date and value columns
Choose models: Select the forecasting methods to apply
Adjust parameters: Configure hyperparameters for each model
Run forecasts: Visualize and compare results
Export results: Download charts and predictions

# Project Structure
time-series-forecaster/
│
├── app_tsf.py          # Main Streamlit application
├── README.md              # This file
└── requirements.txt       # Dependencies
Data Format
Data should be in CSV format with at least two columns:


# Technologies Used

Streamlit: Web application framework
Pandas/NumPy: Data manipulation and analysis
Matplotlib/Seaborn: Data visualization
Statsmodels: Statistical models for time series
LightGBM/XGBoost: Gradient boosting algorithms
TensorFlow/Keras: Deep neural networks
Scikit-learn: Machine learning tools

# Contributing
Contributions are welcome! Please:

Fork the project
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

# Support
If you encounter any issues or have suggestions, please open an issue in the repository.