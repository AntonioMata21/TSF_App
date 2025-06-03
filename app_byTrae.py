import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import lightgbm as lgb
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import io
from io import BytesIO
import base64
import zipfile
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Time Series Forecasting App",
    page_icon="📈",
    layout="wide"
)

# Application title and description
st.title("Time Series Forecasting Application")
st.markdown("""
This application helps you analyze time series data, build forecasting models, 
and evaluate their performance. Upload your data, select models, and get predictions!
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Upload", "Data Preprocessing", "Model Selection", "Forecasting", "Model Comparison", "Export Results"])

# Function to load data
def load_data(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to preprocess data
def preprocess_data(df, date_col, value_col):
    try:
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(by=date_col)
        
        # Set date as index
        df = df.set_index(date_col)
        
        # Check for missing values
        if df[value_col].isnull().sum() > 0:
            st.warning(f"Found {df[value_col].isnull().sum()} missing values. Interpolating...")
            df[value_col] = df[value_col].interpolate(method='linear')
        
        return df
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None

# Function to check stationarity
def check_stationarity(series):
    result = adfuller(series.dropna())
    st.write('ADF Statistic: %f' % result[0])
    st.write('p-value: %f' % result[1])
    st.write('Critical Values:')
    for key, value in result[4].items():
        st.write('\t%s: %.3f' % (key, value))
    
    # Interpret results
    if result[1] <= 0.05:
        st.success("Series is stationary (p-value <= 0.05)")
    else:
        st.warning("Series is not stationary (p-value > 0.05)")

# Function to train ARIMA model
def train_arima(data, p, d, q, forecast_horizon=30):
    # Split data into train and test using forecast_horizon
    train_size = len(data) - forecast_horizon
    train, test = data[:train_size], data[train_size:]
    
    # Train ARIMA model
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()
    
    # Make predictions
    predictions = model_fit.forecast(steps=len(test))
    
    return train, test, predictions, model_fit

# Function to train LightGBM model
def train_lightgbm(data, target_col, features, forecast_horizon=30):
    # Crear una copia del dataframe
    df = data.copy()
    
    # Usar el horizonte de pronóstico como tamaño del conjunto de prueba
    train_size_int = len(df) - forecast_horizon
    train_data = df[:train_size_int].copy()
    test_data = df[train_size_int:].copy()
    
    # Crear características de lag SOLO en los datos de entrenamiento
    for lag in range(1, features + 1):
        train_data[f'lag_{lag}'] = train_data[target_col].shift(lag)
    
    # Eliminar filas con valores NaN
    train_data = train_data.dropna()
    
    # Preparar X_train e y_train
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    
    # Entrenar el modelo LightGBM
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 100
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    # Preparar datos de prueba con características de lag
    # Importante: usar solo datos disponibles hasta el momento de la predicción
    X_test_list = []
    y_test = test_data[target_col].values
    
    # Para cada punto en el conjunto de prueba, crear características usando solo datos pasados
    current_data = train_data.iloc[-features:].copy()  # Últimas filas del conjunto de entrenamiento
    
    for i in range(len(test_data)):
        # Obtener el valor actual
        current_value = test_data.iloc[i][target_col]
        
        # Crear una fila para predecir el valor actual
        test_row = pd.DataFrame(index=[0])
        for lag in range(1, features + 1):
            if lag <= len(current_data):
                test_row[f'lag_{lag}'] = current_data.iloc[-lag][target_col]
            else:
                test_row[f'lag_{lag}'] = np.nan
        
        # Guardar la fila para predicción
        X_test_list.append(test_row)
        
        # Actualizar datos actuales para la siguiente predicción
        new_row = pd.DataFrame({target_col: [current_value]}, index=[current_data.index.max() + pd.Timedelta(days=1)])
        current_data = pd.concat([current_data, new_row]).iloc[1:]
    
    # Convertir lista a DataFrame
    X_test = pd.concat(X_test_list, ignore_index=True)
    
    # Hacer predicciones en el conjunto de prueba
    predictions = model.predict(X_test)
    
    return y_train, pd.Series(y_test, index=test_data.index), predictions, model

    # Create features (lags, etc.)
    df = data.copy()
    
    # Add lag features
    for lag in range(1, features + 1):
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Split data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train-test split
    train_size = int(len(df) * train_size)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train LightGBM model
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 100
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Removed future forecasting functionality
    
    return y_train, y_test, predictions, None, model

# Function to train Simple Moving Average (SMA) model
def train_sma(data, value_col, forecast_horizon, window):
    train_size_int = len(data) - forecast_horizon
    train_data = data.iloc[:train_size_int]
    test_data = data.iloc[train_size_int:]

    # Simple Moving Average (SMA)
    sma_train_pred = train_data[value_col].rolling(window=window).mean().dropna()
    
    # For simplicity, we'll use the last SMA value to forecast the test set and future
    if not sma_train_pred.empty:
        last_sma_value = sma_train_pred.iloc[-1]
    else:
        last_sma_value = train_data[value_col].mean() # Fallback if SMA cannot be calculated

    predictions = pd.Series([last_sma_value] * len(test_data), index=test_data.index)
    
    return train_data[value_col], test_data[value_col], predictions, None # No model object to return for SMA

# Function to train Exponential Smoothing (ETS) model
def train_ets(data, trend, seasonal, seasonal_periods, forecast_horizon=30):
    train_size_int = len(data) - forecast_horizon
    train, test = data[:train_size_int], data[train_size_int:]
    
    model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    model_fit = model.fit()
    
    predictions = model_fit.forecast(steps=len(test))
    
    return train, test, predictions, model_fit

# Function to train XGBoost model
def train_xgboost(data, target_col, features=10, n_estimators=100, max_depth=6,
                  learning_rate=0.1, forecast_horizon=30):
        
    # Validaciones
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data debe ser un pandas DataFrame")
    
    if target_col not in data.columns:
        raise ValueError(f"La columna '{target_col}' no existe en los datos")
    
    if features is None or features <= 0:
        features = 10
    features = int(features)
    
    # Removed train_size validation
    
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon debe ser positivo")
    
    if len(data) < features + 10:  # Mínimo de datos necesarios
        raise ValueError(f"Se necesitan al menos {features + 10} observaciones")
    
    # Crear copia de los datos
    df = data.copy()
    
    # Verificar que target_col sea numérico
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(f"La columna '{target_col}' debe ser numérica")
    
    # Crear características lag (solo una vez)
    print(f"Creando {features} características lag...")
    for lag in range(1, features + 1):
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Eliminar filas con valores faltantes
    df_clean = df.dropna().copy()
    
    if len(df_clean) == 0:
        raise ValueError("No hay datos suficientes después de crear las características lag")
    
    print(f"Datos disponibles después de crear lags: {len(df_clean)}")
    
    # División temporal de datos (importante para series temporales)
    train_size_int = len(df_clean) - forecast_horizon
    
    if train_size_int < features:
        raise ValueError("Datos de entrenamiento insuficientes")
    
    train_data = df_clean[:train_size_int].copy()
    test_data = df_clean[train_size_int:].copy()
    
    # Preparar características y objetivo
    feature_cols = [col for col in df_clean.columns if col.startswith('lag_')]
    
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]
    
    print(f"Datos de entrenamiento: {len(X_train)}")
    print(f"Datos de prueba: {len(X_test)}")
    print(f"Características utilizadas: {feature_cols}")
    
    # Entrenar modelo XGBoost
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predicciones en conjunto de prueba
    if len(X_test) > 0:
        predictions = model.predict(X_test)
        predictions_series = pd.Series(predictions, index=X_test.index)
    else:
        predictions_series = pd.Series(dtype=float)
        print("Advertencia: No hay datos de prueba")
    
    print("Entrenamiento completado exitosamente")
    
    return y_train, y_test, predictions_series, model

# Function to train RNN model
def train_rnn(data, target_col, features=10, units=50, epochs=50, batch_size=32, forecast_horizon=30):
    if features is None:
        features = 10 # Default value if features is None
    features = int(features) # Ensure features is an integer

    if units is None:
        units = 50 # Default value if units is None
    if epochs is None:
        epochs = 50 # Default value if epochs is None
    if batch_size is None:
        batch_size = 32 # Default value if batch_size is None
    df = data.copy()
    
    # Normalize data
    scaler = StandardScaler()
    df[target_col] = scaler.fit_transform(df[[target_col]])
    
    # Create sequences
    X, y = [], []
    for i in range(len(df) - features):
        X.append(df[target_col].iloc[i:(i + features)].values)
        y.append(df[target_col].iloc[i + features])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for RNN [samples, timesteps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Use forecast_horizon as test set size
    train_size_int = len(X) - forecast_horizon
    X_train, X_test = X[:train_size_int], X[train_size_int:]
    y_train, y_test = y[:train_size_int], y[train_size_int:]
    
    model = Sequential([
        SimpleRNN(units, activation='relu', input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    
    predictions = model.predict(X_test)
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    # Convert predictions to Pandas Series with the correct index
    predictions_series = pd.Series(predictions, index=data.index[features + len(y_train) : features + len(y_train) + len(y_test)])

    return (pd.Series(scaler.inverse_transform(y_train.reshape(-1, 1)).flatten(), index=data.index[features : features + len(y_train)]),
            pd.Series(scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), index=data.index[features + len(y_train) : features + len(y_train) + len(y_test)]),
            predictions_series, model)

# Function to train LSTM model
def train_lstm(data, target_col, features=10, units=50, epochs=50, batch_size=32, forecast_horizon=30):
    if features is None:
        features = 10 # Default value if features is None
    features = int(features) # Ensure features is an integer

    if units is None:
        units = 50 # Default value if units is None
    if epochs is None:
        epochs = 50 # Default value if epochs is None
    if batch_size is None:
        batch_size = 32 # Default value if batch_size is None
    df = data.copy()
    
    # Normalize data
    scaler = StandardScaler()
    df[target_col] = scaler.fit_transform(df[[target_col]])
    
    # Create sequences
    X, y = [], []
    for i in range(len(df) - features):
        X.append(df[target_col].iloc[i:(i + features)].values)
        y.append(df[target_col].iloc[i + features])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for LSTM [samples, timesteps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Use forecast_horizon as test set size
    train_size_int = len(X) - forecast_horizon
    X_train, X_test = X[:train_size_int], X[train_size_int:]
    y_train, y_test = y[:train_size_int], y[train_size_int:]
    
    model = Sequential([
        LSTM(units, activation='relu', input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    
    predictions = model.predict(X_test)
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()





    
    # Convert predictions to Pandas Series with the correct index
    predictions_series = pd.Series(predictions, index=data.index[features + len(y_train) : features + len(y_train) + len(y_test)])

    return (pd.Series(scaler.inverse_transform(y_train.reshape(-1, 1)).flatten(), index=data.index[features : features + len(y_train)]),
            pd.Series(scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), index=data.index[features + len(y_train) : features + len(y_train) + len(y_test)]),
            predictions_series, model)

def train_random_forest(data, target_col, features, n_estimators=100, max_depth=10, forecast_horizon=30):
    from sklearn.ensemble import RandomForestRegressor
    
    # Crear una copia del dataframe
    df = data.copy()
    
    # Usar el horizonte de pronóstico como tamaño del conjunto de prueba
    train_size_int = len(df) - forecast_horizon
    train_data = df[:train_size_int].copy()
    test_data = df[train_size_int:].copy()
    
    # Crear características de lag SOLO en los datos de entrenamiento
    for lag in range(1, features + 1):
        train_data[f'lag_{lag}'] = train_data[target_col].shift(lag)
    
    # Eliminar filas con valores NaN
    train_data = train_data.dropna()
    
    # Preparar X_train e y_train
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    
    # Entrenar el modelo Random Forest
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Preparar datos de prueba con características de lag
    X_test_list = []
    y_test = test_data[target_col].values
    
    # Para cada punto en el conjunto de prueba, crear características usando solo datos pasados
    current_data = train_data.iloc[-features:].copy()  # Últimas filas del conjunto de entrenamiento
    
    for i in range(len(test_data)):
        # Obtener el valor actual
        current_value = test_data.iloc[i][target_col]
        
        # Crear una fila para predecir el valor actual
        test_row = pd.DataFrame(index=[0])
        for lag in range(1, features + 1):
            if lag <= len(current_data):
                test_row[f'lag_{lag}'] = current_data.iloc[-lag][target_col]
            else:
                test_row[f'lag_{lag}'] = np.nan
        
        # Guardar la fila para predicción
        X_test_list.append(test_row)
        
        # Actualizar datos actuales para la siguiente predicción
        new_row = pd.DataFrame({target_col: [current_value]}, index=[current_data.index.max() + pd.Timedelta(days=1)])
        current_data = pd.concat([current_data, new_row]).iloc[1:]
    
    # Convertir lista a DataFrame
    X_test = pd.concat(X_test_list, ignore_index=True)
    
    # Hacer predicciones en el conjunto de prueba
    predictions = model.predict(X_test)
    
    return y_train, pd.Series(y_test, index=test_data.index), predictions, model
def aggregate_time_series(data, value_col, freq='D', method='mean'):
    """
    Agrega los datos de series temporales según la frecuencia y método especificados.
    
    Parámetros:
    - data: DataFrame con índice de fecha
    - value_col: Columna con los valores a agregar
    - freq: Frecuencia de agregación ('D': diario, 'W': semanal, 'M': mensual, 'Q': trimestral, 'Y': anual)
    - method: Método de agregación ('mean', 'sum', 'min', 'max', 'median')
    
    Retorna:
    - DataFrame agregado
    """
    if method == 'mean':
        return data[value_col].resample(freq).mean()
    elif method == 'sum':
        return data[value_col].resample(freq).sum()
    elif method == 'min':
        return data[value_col].resample(freq).min()
    elif method == 'max':
        return data[value_col].resample(freq).max()
    elif method == 'median':
        return data[value_col].resample(freq).median()
    else:
        return data[value_col].resample(freq).mean()

# Añadir una función para visualizar datos de series temporales
def plot_time_series(data, value_col, title='Time Series Plot', figsize=(12, 6)):
    """
    Crea una visualización de series temporales.
    
    Parámetros:
    - data: DataFrame con índice de fecha
    - value_col: Columna con los valores a visualizar
    - title: Título del gráfico
    - figsize: Tamaño de la figura
    
    Retorna:
    - fig, ax: Objetos de figura y ejes de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data.index, data[value_col])
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel(value_col)
    plt.tight_layout()
    return fig, ax
# Function to calculate metrics
def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

# Function to create downloadable link
def get_download_link(df, filename, text):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to save plot as image
def get_image_download_link(fig, filename, text):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

def convert_df_to_csv(df):
    """Convert a DataFrame to CSV for download"""
    return df.to_csv(index=True).encode('utf-8')

# Data Upload Page
if page == "Data Upload":
    st.header("Upload Your Time Series Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.session_state['data'] = df
            st.success("Data loaded successfully!")
            
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            st.subheader("Data Information")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
            
            st.subheader("Select Date and Value Columns")
            date_col = st.selectbox("Select Date Column", df.columns)
            value_col = st.selectbox("Select Value Column", df.columns)
            
            if st.button("Confirm Columns"):
                st.session_state['date_col'] = date_col
                st.session_state['value_col'] = value_col
                st.success(f"Selected Date Column: {date_col}, Value Column: {value_col}")

# Data Preprocessing Page
elif page == "Data Preprocessing":
    st.header("Data Preprocessing")
    
    if 'data' not in st.session_state or 'date_col' not in st.session_state or 'value_col' not in st.session_state:
        st.warning("Please upload data and select columns first!")
    else:
        df = st.session_state['data']
        date_col = st.session_state['date_col']
        value_col = st.session_state['value_col']
        
        st.subheader("Original Data")
        st.dataframe(df.head())
        
        # Preprocess data
        processed_df = preprocess_data(df, date_col, value_col)
        
        if processed_df is not None:
            st.session_state['processed_data'] = processed_df
            
            st.subheader("Processed Data")
            st.dataframe(processed_df.head())
            
            # Agregación de series temporales
            st.subheader("Time Series Aggregation")
            st.write("Aggregate your time series data to different frequencies:")
            
            col1, col2 = st.columns(2)
            with col1:
                agg_freq = st.selectbox(
                    "Select aggregation frequency:",
                    ["Daily (D)", "Weekly (W)", "Monthly (M)", "Quarterly (Q)", "Yearly (Y)"]
                )
            with col2:
                agg_method = st.selectbox(
                    "Select aggregation method:",
                    ["Mean", "Sum", "Min", "Max", "Median"]
                )
            
            # Mapear selecciones a parámetros
            freq_map = {
                "Daily (D)": "D",
                "Weekly (W)": "W",
                "Monthly (M)": "M",
                "Quarterly (Q)": "Q",
                "Yearly (Y)": "Y"
            }
            
            method_map = {
                "Mean": "mean",
                "Sum": "sum",
                "Min": "min",
                "Max": "max",
                "Median": "median"
            }
            
            # Aplicar agregación
            if st.button("Apply Aggregation"):
                freq = freq_map[agg_freq]
                method = method_map[agg_method]
                
                aggregated_data = aggregate_time_series(processed_df, value_col, freq, method)
                st.session_state['aggregated_data'] = aggregated_data
                
                # Mostrar datos agregados
                st.subheader(f"Aggregated Data ({agg_freq}, {agg_method})")
                st.dataframe(aggregated_data.head())
                
                # Visualizar datos agregados
                fig, ax = plot_time_series(
                    pd.DataFrame(aggregated_data), 
                    aggregated_data.name, 
                    title=f"Aggregated Time Series ({agg_freq}, {agg_method})"
                )
                st.pyplot(fig)
                
                # Guardar en session_state para uso posterior
                st.session_state['current_data'] = pd.DataFrame(aggregated_data)
                st.session_state['current_data_name'] = f"Aggregated ({agg_freq}, {agg_method})"
                
                # Opción para usar datos agregados
                st.session_state['use_aggregated_data_for_forecast'] = st.checkbox("Use aggregated data for forecasting", value=st.session_state.get('use_aggregated_data_for_forecast', False))
                if st.session_state['use_aggregated_data_for_forecast']:
                    st.success(f"Aggregated data will be used for forecasting!")
                else:
                    st.info(f"Original processed data will be used for forecasting!")
            
            # Visualización de datos originales
            st.subheader("Original Time Series Visualization")
            fig_orig, ax_orig = plot_time_series(processed_df, value_col, title=f"Original Time Series: {value_col}")
            st.pyplot(fig_orig)
            
            # Check stationarity
            st.subheader("Stationarity Check")
            check_stationarity(processed_df[value_col])
            
            # ACF and PACF plots
            st.subheader("ACF and PACF Plots")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            plot_acf(processed_df[value_col].dropna(), ax=ax1)
            plot_pacf(processed_df[value_col].dropna(), ax=ax2)
            st.pyplot(fig)
            
            # Seasonal decomposition
            st.subheader("Seasonal Decomposition")
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                decomposition = seasonal_decompose(processed_df[value_col], model='additive', period=st.slider("Select period for decomposition", 1, 52, 12))
                
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
                decomposition.observed.plot(ax=ax1)
                ax1.set_title('Observed')
                decomposition.trend.plot(ax=ax2)
                ax2.set_title('Trend')
                decomposition.seasonal.plot(ax=ax3)
                ax3.set_title('Seasonality')
                decomposition.resid.plot(ax=ax4)
                ax4.set_title('Residuals')
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error in seasonal decomposition: {e}")

# Model Selection Page
elif page == "Model Selection":
    st.header("Model Selection")
    
    if 'processed_data' not in st.session_state:
        st.warning("Please preprocess your data first!")
    else:
        processed_data = st.session_state['processed_data']
        value_col = st.session_state['value_col']
        
        st.subheader("Select Forecasting Models")
        
        # Initialize active_models in session_state if not present
        if 'active_models' not in st.session_state:
            st.session_state['active_models'] = []

        available_models = {
            "ARIMA": "ARIMA",
            "LightGBM": "LightGBM",
            "Exponential Smoothing": "ETS",
            "Random Forest": "RF",
            "Simple Moving Average (SMA)": "SMA",
            "XGBoost": "XGBoost",
            "Recurrent Neural Network (RNN)": "RNN",
            "Long Short-Term Memory (LSTM)": "LSTM"
        }

        selected_model_name = st.selectbox(
            "Choose a model to add:",
            list(available_models.keys())
        )

        if st.button("Add Model to Comparison"):
            if len(st.session_state['active_models']) < 2:
                if selected_model_name not in [m['name'] for m in st.session_state['active_models']]:
                    st.session_state['active_models'].append({'name': selected_model_name, 'params': {}})
                    st.success(f"{selected_model_name} added for comparison.")
                else:
                    st.warning(f"{selected_model_name} is already in the comparison list.")
            else:
                st.warning("You can only compare a maximum of two models.")

        st.subheader("Models Selected for Comparison:")
        if st.session_state['active_models']:
            for i, model_info in enumerate(st.session_state['active_models']):
                st.write(f"- {model_info['name']}")
                if st.button(f"Remove {model_info['name']}", key=f"remove_{model_info['name']}"):
                    st.session_state['active_models'].pop(i)
                    st.experimental_rerun()
        else:
            st.info("No models selected for comparison yet.")

        # Model Parameters based on selected models
        for model_info in st.session_state['active_models']:
            if model_info['name'] == "ARIMA":
                st.subheader("ARIMA Model Parameters")
                col1, col2, col3 = st.columns(3)
                with col1:
                    p = st.slider("p (AR order)", 0, 5, 1, key="arima_p")
                with col2:
                    d = st.slider("d (Differencing)", 0, 2, 1, key="arima_d")
                with col3:
                    q = st.slider("q (MA order)", 0, 5, 1, key="arima_q")
                model_info['params'] = {'p': p, 'd': d, 'q': q}

            elif model_info['name'] == "LightGBM":
                st.subheader("LightGBM Model Parameters")
                num_features = st.slider("Number of lag features", 1, 30, 7, key="lgbm_features")
                model_info['params'] = {'num_features': num_features}

            elif model_info['name'] == "Exponential Smoothing":
                st.subheader("Exponential Smoothing Model Parameters")
                col1, col2, col3 = st.columns(3)
                with col1:
                    ets_trend = st.selectbox("Trend", [None, 'add', 'mul'], key="ets_trend")
                with col2:
                    ets_seasonal = st.selectbox("Seasonal", [None, 'add', 'mul'], key="ets_seasonal")
                with col3:
                    ets_seasonal_periods = st.slider("Seasonal Periods", 1, 365, 7, key="ets_seasonal_periods")
                model_info['params'] = {'trend': ets_trend, 'seasonal': ets_seasonal, 'seasonal_periods': ets_seasonal_periods}

            elif model_info['name'] == "Simple Moving Average (SMA)":
                st.subheader("Simple Moving Average (SMA) Model Parameters")
                sma_window = st.slider("Window Size", 1, 30, 7, key="sma_window")
                model_info['params'] = {'window': sma_window}

            elif model_info['name'] == "XGBoost":
                st.subheader("XGBoost Model Parameters")
                col1, col2, col3 = st.columns(3)
                with col1:
                    xgb_n_estimators = st.slider("Number of Estimators", 50, 500, 100, key="xgb_n_estimators")
                with col2:
                    xgb_max_depth = st.slider("Max Depth", 1, 10, 6, key="xgb_max_depth")
                with col3:
                    xgb_learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01, key="xgb_learning_rate")
                xgb_features = st.slider("Number of lag features", 1, 30, 7, key="xgb_features")
                model_info['params'] = {'n_estimators': xgb_n_estimators, 'max_depth': xgb_max_depth, 'learning_rate': xgb_learning_rate, 'features': xgb_features}

            elif model_info['name'] == "Recurrent Neural Network (RNN)":
                st.subheader("RNN Model Parameters")
                col1, col2, col3 = st.columns(3)
                with col1:
                    rnn_features = st.slider("Number of lag features", 1, 30, 7, key="rnn_features")
                with col2:
                    rnn_units = st.slider("Units", 10, 100, 50, key="rnn_units")
                with col3:
                    rnn_epochs = st.slider("Epochs", 10, 200, 50, key="rnn_epochs")
                rnn_batch_size = st.slider("Batch Size", 16, 128, 32, key="rnn_batch_size")
                model_info['params'] = {'features': rnn_features, 'units': rnn_units, 'epochs': rnn_epochs, 'batch_size': rnn_batch_size}

            elif model_info['name'] == "Long Short-Term Memory (LSTM)":
                st.subheader("LSTM Model Parameters")
                col1, col2, col3 = st.columns(3)
                with col1:
                    lstm_features = st.slider("Number of lag features", 1, 30, 7, key="lstm_features")
                with col2:
                    lstm_units = st.slider("Units", 10, 100, 50, key="lstm_units")
                with col3:
                    lstm_epochs = st.slider("Epochs", 10, 200, 50, key="lstm_epochs")
                lstm_batch_size = st.slider("Batch Size", 16, 128, 32, key="lstm_batch_size")
                model_info['params'] = {'features': lstm_features, 'units': lstm_units, 'epochs': lstm_epochs, 'batch_size': lstm_batch_size}
                trend_options = [None, 'add', 'mul']
                seasonal_options = [None, 'add', 'mul']
                seasonal_periods = st.slider("Seasonal Periods", 1, 30, 7, key="ets_seasonal_periods")
                trend = st.selectbox("Trend", trend_options, key="ets_trend")
                seasonal = st.selectbox("Seasonal", seasonal_options, key="ets_seasonal")
                model_info['params'] = {
                    'seasonal_periods': seasonal_periods,
                    'trend': trend,
                    'seasonal': seasonal
                }

            elif model_info['name'] == "Random Forest":
                st.subheader("Random Forest Model Parameters")
                n_estimators = st.slider("Number of Estimators", 50, 500, 100, key="rf_n_estimators")
                max_depth = st.slider("Max Depth", 1, 20, 10, key="rf_max_depth")
                num_lag_features_rf = st.slider("Number of lag features (RF)", 1, 30, 7, key="rf_lag_features")
                model_info['params'] = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'num_lag_features': num_lag_features_rf
                }

        # Nota: El conjunto de entrenamiento se define automáticamente como los datos anteriores al horizonte de pronóstico
        st.session_state['train_size'] = None  # Ya no se usa, pero mantenemos la clave para compatibilidad
        
        # Forecast horizon
        st.subheader("Forecast Horizon")
        forecast_horizon = st.slider("Number of periods to forecast", 1, 100, 30)
        st.session_state['forecast_horizon'] = forecast_horizon

        # Time Series Trimming
        st.subheader("Time Series Trimming")
        trim_option = st.radio(
            "Trim time series data?",
            ("No trimming", "Trim by last X years", "Trim by last X observations")
        )
        st.session_state['trim_option'] = trim_option
        
        if trim_option == "Trim by last X years":
            years_to_keep = st.number_input("Number of last years to keep:", min_value=1, value=1)
            st.session_state['trim_value'] = {'unit': 'years', 'value': years_to_keep}
            
            # Vista previa del recorte
            if st.button("Preview Trimming"):
                cutoff_date = processed_data.index.max() - pd.DateOffset(years=years_to_keep)
                trimmed_data = processed_data[processed_data.index >= cutoff_date]
                
                # Visualizar datos originales vs recortados
                st.subheader("Trimming Preview")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(processed_data.index, processed_data[value_col], label='Original Data', alpha=0.5)
                ax.plot(trimmed_data.index, trimmed_data[value_col], label=f'Trimmed Data (Last {years_to_keep} years)', color='red')
                ax.axvline(x=cutoff_date, color='green', linestyle='--', label='Cutoff Date')
                ax.set_title(f'Time Series Trimming Preview')
                ax.set_xlabel('Date')
                ax.set_ylabel(value_col)
                ax.legend()
                st.pyplot(fig)
                
                # Mostrar estadísticas
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Data Points", len(processed_data))
                    st.metric("Original Date Range", f"{processed_data.index.min().date()} to {processed_data.index.max().date()}")
                with col2:
                    st.metric("Trimmed Data Points", len(trimmed_data))
                    st.metric("Trimmed Date Range", f"{trimmed_data.index.min().date()} to {trimmed_data.index.max().date()}")
                    
                # Guardar datos recortados temporalmente
                st.session_state['preview_trimmed_data'] = trimmed_data
                
        elif trim_option == "Trim by last X observations":
            obs_to_keep = st.number_input("Number of last observations to keep:", min_value=1, value=100)
            st.session_state['trim_value'] = {'unit': 'observations', 'value': obs_to_keep}
            
            # Vista previa del recorte
            if st.button("Preview Trimming"):
                if obs_to_keep < len(processed_data):
                    trimmed_data = processed_data.iloc[-obs_to_keep:]
                    
                    # Visualizar datos originales vs recortados
                    st.subheader("Trimming Preview")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(processed_data.index, processed_data[value_col], label='Original Data', alpha=0.5)
                    ax.plot(trimmed_data.index, trimmed_data[value_col], label=f'Trimmed Data (Last {obs_to_keep} observations)', color='red')
                    ax.axvline(x=trimmed_data.index[0], color='green', linestyle='--', label='Cutoff Point')
                    ax.set_title(f'Time Series Trimming Preview')
                    ax.set_xlabel('Date')
                    ax.set_ylabel(value_col)
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Mostrar estadísticas
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Original Data Points", len(processed_data))
                        st.metric("Original Date Range", f"{processed_data.index.min().date()} to {processed_data.index.max().date()}")
                    with col2:
                        st.metric("Trimmed Data Points", len(trimmed_data))
                        st.metric("Trimmed Date Range", f"{trimmed_data.index.min().date()} to {trimmed_data.index.max().date()}")
                        
                    # Guardar datos recortados temporalmente
                    st.session_state['preview_trimmed_data'] = trimmed_data
                else:
                    st.warning(f"Number of observations to keep ({obs_to_keep}) is greater than or equal to the total number of observations ({len(processed_data)}). No trimming will be applied.")
        else:
            st.session_state['trim_value'] = None

        # Save model selection (now just a confirmation button)
        if st.button("Confirm Model Selection and Parameters"):
            st.success("Model selection and parameters saved!")

# Forecasting Page
elif page == "Forecasting":
    st.header("Forecasting")
    
    if 'processed_data' not in st.session_state or 'train_size' not in st.session_state or 'active_models' not in st.session_state:
        st.warning("Please select models first!")
    else:
        processed_data_original = st.session_state['processed_data']
        value_col = st.session_state['value_col']
        train_size = st.session_state['train_size']
        forecast_horizon = st.session_state.get('forecast_horizon', 30)

        # Determine which data to use for forecasting
        if st.session_state.get('use_aggregated_data_for_forecast', False) and 'aggregated_data' in st.session_state:
            processed_data_for_forecast = pd.DataFrame(st.session_state['aggregated_data'])
            processed_data_for_forecast.columns = [value_col] # Ensure column name consistency
            st.subheader("Data for Forecasting (Aggregated)")
        else:
            processed_data_for_forecast = processed_data_original
            st.subheader("Data for Forecasting (Original Processed)")
        
        # Apply trimming if selected
        if st.session_state.get('trim_option', "No trimming") != "No trimming" and st.session_state.get('trim_value') is not None:
            trim_info = st.session_state['trim_value']
            if trim_info['unit'] == 'years':
                cutoff_date = processed_data_for_forecast.index.max() - pd.DateOffset(years=trim_info['value'])
                trimmed_data = processed_data_for_forecast[processed_data_for_forecast.index >= cutoff_date]
                st.info(f"Data trimmed to last {trim_info['value']} years (from {cutoff_date.date()} to {processed_data_for_forecast.index.max().date()})")
                
                # Visualizar datos recortados
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(processed_data_for_forecast.index, processed_data_for_forecast[value_col], label='Original Data', alpha=0.3)
                ax.plot(trimmed_data.index, trimmed_data[value_col], label='Trimmed Data (Used for Forecasting)', color='blue')
                ax.axvline(x=cutoff_date, color='red', linestyle='--', label='Cutoff Date')
                ax.set_title('Data Used for Forecasting')
                ax.set_xlabel('Date')
                ax.set_ylabel(value_col)
                ax.legend()
                st.pyplot(fig)
                
                # Actualizar datos para pronóstico
                processed_data_for_forecast = trimmed_data
                
            elif trim_info['unit'] == 'observations':
                trimmed_data = processed_data_for_forecast.iloc[-trim_info['value']:]
                st.info(f"Data trimmed to last {trim_info['value']} observations")
                
                # Visualizar datos recortados
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(processed_data_for_forecast.index, processed_data_for_forecast[value_col], label='Original Data', alpha=0.3)
                ax.plot(trimmed_data.index, trimmed_data[value_col], label='Trimmed Data (Used for Forecasting)', color='blue')
                ax.axvline(x=trimmed_data.index[0], color='red', linestyle='--', label='Cutoff Point')
                ax.set_title('Data Used for Forecasting')
                ax.set_xlabel('Date')
                ax.set_ylabel(value_col)
                ax.legend()
                st.pyplot(fig)
                
                # Actualizar datos para pronóstico
                processed_data_for_forecast = trimmed_data
        else:
            # Visualizar datos completos
            fig, ax = plot_time_series(processed_data_for_forecast, value_col, title='Complete Data Used for Forecasting')
            st.pyplot(fig)
        
        # Mostrar información sobre el conjunto de datos
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Points", len(processed_data_for_forecast))
        with col2:
            st.metric("Date Range", f"{processed_data_for_forecast.index.min().date()} to {processed_data_for_forecast.index.max().date()}")
        with col3:
            train_size_int = len(processed_data_for_forecast) - forecast_horizon
            st.metric("Train-Test Split", f"{train_size_int} : {forecast_horizon}")
        
        # Mostrar modelos seleccionados
        st.subheader("Selected Models")
        for model_info in st.session_state['active_models']:
            st.write(f"- **{model_info['name']}** with parameters: {model_info['params']}")
        
        # Train models and make predictions
        if st.button("Run Forecasting"):
            results = {}
            
            # Process each active model
            for model_info in st.session_state['active_models']:
                model_name = model_info['name']
                model_params = model_info['params']
                
                # ARIMA model
                if model_name == "ARIMA":
                    with st.spinner("Training ARIMA model..."):
                        p = model_params.get('p', 1)
                        d = model_params.get('d', 1)
                        q = model_params.get('q', 1)
                        train, test, predictions, model = train_arima(
                            processed_data_for_forecast[value_col], p, d, q, forecast_horizon
                        )
                        
                        # Calculate metrics
                        metrics = calculate_metrics(test.values, predictions)
                        
                        results['ARIMA'] = {
                            'train': train,
                            'test': test,
                            'predictions': predictions,
                            'metrics': metrics,
                            'model': model
                        }
                        
                        st.success("ARIMA model trained successfully!")
                        
                        # Display metrics
                        st.subheader("ARIMA Model Metrics")
                        metrics_df = pd.DataFrame([metrics])
                        st.dataframe(metrics_df)
                        
                        # Plot results with interactive elements
                        st.subheader("ARIMA Forecast")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Checkbox para mostrar/ocultar datos de entrenamiento
                        show_train = st.checkbox("Show Training Data", value=True, key="arima_show_train")
                        if show_train:
                            ax.plot(train.index, train, label='Training Data', alpha=0.5)
                        
                        ax.plot(test.index, test, label='Actual Values')
                        ax.plot(test.index, predictions, label='Predictions', color='red')
                        ax.set_title(f'ARIMA({p},{d},{q}) Forecast')
                        ax.legend()
                        st.pyplot(fig)
                
                # LightGBM model
                elif model_name == "LightGBM":
                    with st.spinner("Training LightGBM model..."):
                        num_features = model_params.get('num_features', 7)
                        
                        y_train, y_test, predictions, model = train_lightgbm(
                            processed_data_for_forecast[[value_col]], value_col, num_features, forecast_horizon
                        )
                        
                        # Calculate metrics
                        metrics = calculate_metrics(y_test.values, predictions)
                        
                        results['LightGBM'] = {
                            'train': y_train,
                            'test': y_test,
                            'predictions': predictions,
                            'metrics': metrics,
                            'model': model
                        }
                        
                        st.success("LightGBM model trained successfully!")
                        
                        # Display metrics
                        st.subheader("LightGBM Model Metrics")
                        metrics_df = pd.DataFrame([metrics])
                        st.dataframe(metrics_df)
                        
                        # Plot results with interactive elements
                        st.subheader("LightGBM Forecast")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Checkbox para mostrar/ocultar datos de entrenamiento
                        show_train = st.checkbox("Show Training Data", value=True, key="lgbm_show_train")
                        if show_train:
                            ax.plot(y_train.index, y_train, label='Training Data', alpha=0.5)
                        
                        ax.plot(y_test.index, y_test, label='Actual Values')
                        ax.plot(y_test.index, predictions, label='Predictions', color='red')
                        
                        # Removed future predictions checkbox and plotting
                        
                        ax.set_title('LightGBM Forecast')
                        ax.legend()
                        st.pyplot(fig)
                
                # Exponential Smoothing model
                elif model_name == "Exponential Smoothing":
                    with st.spinner("Training Exponential Smoothing model..."):
                        from statsmodels.tsa.holtwinters import ExponentialSmoothing
                        
                        # Get parameters
                        seasonal_periods = model_params.get('seasonal_periods', 7)
                        trend = model_params.get('trend', None)
                        seasonal = model_params.get('seasonal', None)
                        
                        # Split data
                        if train_size is None:
                            train_size = 0.8  # Default value if not set
                        train_size_int = int(len(processed_data_for_forecast) * train_size)
                        train = processed_data_for_forecast[value_col][:train_size_int]
                        test = processed_data_for_forecast[value_col][train_size_int:]
                        
                        # Train model
                        model = ExponentialSmoothing(
                            train,
                            trend=trend,
                            seasonal=seasonal,
                            seasonal_periods=seasonal_periods
                        ).fit()
                        
                        # Make predictions
                        predictions = model.forecast(len(test))
                        future_preds = model.forecast(forecast_horizon)
                        
                        # Calculate metrics
                        metrics = calculate_metrics(test.values, predictions)
                        
                        results['ETS'] = {
                            'train': train,
                            'test': test,
                            'predictions': predictions,
                            'metrics': metrics,
                            'model': model
                        }
                        
                        st.success("Exponential Smoothing model trained successfully!")
                        
                        # Display metrics
                        st.subheader("Exponential Smoothing Model Metrics")
                        metrics_df = pd.DataFrame([metrics])
                        st.dataframe(metrics_df)
                        
                        # Plot results with interactive elements
                        st.subheader("Exponential Smoothing Forecast")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Checkbox para mostrar/ocultar datos de entrenamiento
                        show_train = st.checkbox("Show Training Data", value=True, key="ets_show_train")
                        if show_train:
                            ax.plot(train.index, train, label='Training Data', alpha=0.5)
                        
                        ax.plot(test.index, test, label='Actual Values')
                        ax.plot(test.index, predictions, label='Predictions', color='red')
                        
                        # Removed future predictions checkbox and plotting
                        
                        ax.set_title('Exponential Smoothing Forecast')
                        ax.legend()
                        st.pyplot(fig)

                # Simple Moving Average (SMA) model
                elif model_name == "Simple Moving Average (SMA)":
                    with st.spinner("Training SMA model..."):
                        sma_params = model_info['params']
                        y_train, y_test, predictions, model = train_sma(
                            processed_data_for_forecast[[value_col]], value_col, forecast_horizon, sma_params.get('window')
                        )
                        future_preds = None
                        
                        # Calculate metrics
                        metrics = calculate_metrics(y_test.values, predictions)

                        results[model_name] = {
                            'train': y_train,
                            'test': y_test,
                            'predictions': predictions,
                            'metrics': metrics,
                            'model': model
                        }
                        st.success("SMA model trained successfully!")

                        # Display metrics
                        st.subheader("SMA Model Metrics")
                        metrics_df = pd.DataFrame([metrics])
                        st.dataframe(metrics_df)

                        # Plot results with interactive elements
                        st.subheader("SMA Forecast")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Checkbox para mostrar/ocultar datos de entrenamiento
                        show_train = st.checkbox("Show Training Data", value=True, key="sma_show_train")
                        if show_train:
                            ax.plot(y_train.index, y_train, label='Training Data', alpha=0.5)
                        
                        ax.plot(y_test.index, y_test, label='Actual Values')
                        ax.plot(predictions.index, predictions, label='Predictions', color='red')
                        
                        # Removed future predictions checkbox and plotting
                        
                        ax.set_title('Simple Moving Average (SMA) Forecast')
                        ax.legend()
                        st.pyplot(fig)

                # XGBoost model
                elif model_name == "XGBoost":
                    with st.spinner("Training XGBoost model..."):
                        xgb_params = model_info['params']
                        y_train, y_test, predictions, model = train_xgboost(
                            processed_data_for_forecast,
                            value_col,
                            n_estimators=xgb_params.get('n_estimators'),
                            max_depth=xgb_params.get('max_depth'),
                            learning_rate=xgb_params.get('learning_rate'),
                            features=xgb_params.get('features'),
                            forecast_horizon=forecast_horizon
                        )
                        future_preds = None
                        
                        # Calculate metrics
                        metrics = calculate_metrics(y_test.values, predictions)

                        results[model_name] = {
                            'train': y_train,
                            'test': y_test,
                            'predictions': predictions,
                            'metrics': metrics,
                            'model': model
                        }
                        st.success("XGBoost model trained successfully!")

                        # Display metrics
                        st.subheader("XGBoost Model Metrics")
                        metrics_df = pd.DataFrame([metrics])
                        st.dataframe(metrics_df)

                        # Plot results with interactive elements
                        st.subheader("XGBoost Forecast")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Checkbox para mostrar/ocultar datos de entrenamiento
                        show_train = st.checkbox("Show Training Data", value=True, key="xgb_show_train")
                        if show_train:
                            ax.plot(y_train.index, y_train, label='Training Data', alpha=0.5)
                        
                        ax.plot(y_test.index, y_test, label='Actual Values')
                        ax.plot(predictions.index, predictions, label='Predictions', color='red')
                        
                        # Removed future predictions checkbox and plotting
                        
                        ax.set_title('XGBoost Forecast')
                        ax.legend()
                        st.pyplot(fig)

                # Recurrent Neural Network (RNN) model
                elif model_name == "Recurrent Neural Network (RNN)":
                    with st.spinner("Training RNN model..."):
                        rnn_params = model_info['params']
                        y_train, y_test, predictions, model = train_rnn(
                            processed_data_for_forecast,
                            value_col,
                            features=rnn_params.get('features'),
                            units=rnn_params.get('units'),
                            epochs=rnn_params.get('epochs'),
                            batch_size=rnn_params.get('batch_size'),
                            forecast_horizon=forecast_horizon
                        )
                        
                        # Calculate metrics
                        metrics = calculate_metrics(y_test.values, predictions)

                        results[model_name] = {
                            'train': y_train,
                            'test': y_test,
                            'predictions': predictions,
                            'metrics': metrics,
                            'model': model
                        }
                        st.success("RNN model trained successfully!")

                        # Display metrics
                        st.subheader("RNN Model Metrics")
                        metrics_df = pd.DataFrame([metrics])
                        st.dataframe(metrics_df)

                        # Plot results with interactive elements
                        st.subheader("RNN Forecast")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Checkbox para mostrar/ocultar datos de entrenamiento
                        show_train = st.checkbox("Show Training Data", value=True, key="rnn_show_train")
                        if show_train:
                            ax.plot(y_train.index, y_train, label='Training Data', alpha=0.5)
                        
                        ax.plot(y_test.index, y_test, label='Actual Values')
                        ax.plot(predictions.index, predictions, label='Predictions', color='red')
                        
                        # Removed future predictions checkbox and plotting
                        
                        ax.set_title('Recurrent Neural Network (RNN) Forecast')
                        ax.legend()
                        st.pyplot(fig)

                # Long Short-Term Memory (LSTM) model
                elif model_name == "Long Short-Term Memory (LSTM)":
                    with st.spinner("Training LSTM model..."):
                        lstm_params = model_info['params']
                        y_train, y_test, predictions, model = train_lstm(
                            processed_data_for_forecast,
                            value_col,
                            features=lstm_params.get('features'),
                            units=lstm_params.get('units'),
                            epochs=lstm_params.get('epochs'),
                            batch_size=lstm_params.get('batch_size'),
                            forecast_horizon=forecast_horizon
                        )
                        
                        # Calculate metrics
                        metrics = calculate_metrics(y_test.values, predictions)

                        results[model_name] = {
                            'train': y_train,
                            'test': y_test,
                            'predictions': predictions,
                            'metrics': metrics,
                            'model': model
                        }
                        st.success("LSTM model trained successfully!")

                        # Display metrics
                        st.subheader("LSTM Model Metrics")
                        metrics_df = pd.DataFrame([metrics])
                        st.dataframe(metrics_df)

                        # Plot results with interactive elements
                        st.subheader("LSTM Forecast")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Checkbox para mostrar/ocultar datos de entrenamiento
                        show_train = st.checkbox("Show Training Data", value=True, key="lstm_show_train")
                        if show_train:
                            ax.plot(y_train.index, y_train, label='Training Data', alpha=0.5)
                        
                        ax.plot(y_test.index, y_test, label='Actual Values')
                        ax.plot(predictions.index, predictions, label='Predictions', color='red')
                        
                        # Removed future predictions checkbox and plotting
                        
                        ax.set_title('Long Short-Term Memory (LSTM) Forecast')
                        ax.legend()
                        st.pyplot(fig)

                # Random Forest model
                elif model_name == "Random Forest":
                    with st.spinner("Training Random Forest model..."):
                        # Get parameters
                        n_estimators = model_params.get('n_estimators', 100)
                        max_depth = model_params.get('max_depth', 10)
                        num_lag_features = model_params.get('num_lag_features', 7)
                        
                        # Train model using the new function
                        y_train, y_test, predictions, model = train_random_forest(
                            processed_data_for_forecast[[value_col]], 
                            value_col, 
                            num_lag_features, 
                            n_estimators, 
                            max_depth, 
                            forecast_horizon
                        )
                        
                        # Calculate metrics
                        metrics = calculate_metrics(y_test.values, predictions)
                        
                        results['RF'] = {
                            'train': y_train,
                            'test': y_test,
                            'predictions': predictions,
                            'metrics': metrics,
                            'model': model
                        }
                        
                        st.success("Random Forest model trained successfully!")
                        
                        # Display metrics
                        st.subheader("Random Forest Model Metrics")
                        metrics_df = pd.DataFrame([metrics])
                        st.dataframe(metrics_df)
                        
                        # Plot results with interactive elements
                        st.subheader("Random Forest Forecast")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Checkbox para mostrar/ocultar datos de entrenamiento
                        show_train = st.checkbox("Show Training Data", value=True, key="rf_show_train")
                        if show_train:
                            ax.plot(y_train.index, y_train, label='Training Data', alpha=0.5)
                        
                        ax.plot(y_test.index, y_test, label='Actual Values')
                        ax.plot(y_test.index, predictions, label='Predictions', color='red')
                        
                        # Removed future predictions checkbox and plotting
                        
                        ax.set_title('Random Forest Forecast')
                        ax.legend()
                        st.pyplot(fig)
            
            # Save results
            st.session_state['results'] = results

# Model Comparison Page
elif page == "Model Comparison":
    st.header("Model Comparison")
    
    if 'results' not in st.session_state:
        st.warning("Please run forecasting first!")
    else:
        results = st.session_state['results']
        
        # Metrics comparison
        st.subheader("Metrics Comparison")
        
        # Create a dataframe with all metrics
        metrics_data = {}
        for model_name, model_results in results.items():
            metrics_data[model_name] = model_results['metrics']
        
        metrics_df = pd.DataFrame(metrics_data).T
        
        # Add styling to highlight the best model for each metric
        def highlight_min(s):
            is_min = s == s.min()
            return ['background-color: lightgreen' if v else '' for v in is_min]
        
        # Apply styling to all numeric columns
        styled_metrics = metrics_df.style.apply(highlight_min)
        
        # Display the styled dataframe
        st.dataframe(styled_metrics)
        
        # Visual comparison
        st.subheader("Visual Comparison")
        
        # Selección de modelos a comparar
        available_models = list(results.keys())
        selected_models = st.multiselect(
            "Select models to compare",
            available_models,
            default=available_models
        )
        
        if selected_models:
            # Opciones de visualización
            col1, col2 = st.columns(2)
            with col1:
                show_train = st.checkbox("Show Training Data", value=False)
            with col2:
                # Removed future predictions checkbox
                pass
            
            # Crear gráfica de comparación
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Colores para cada modelo
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
            
            # Datos reales (solo necesitamos mostrarlos una vez)
            model_data = results[selected_models[0]]
            test_data = model_data['test']
            ax.plot(test_data.index, test_data, label='Actual Values', color='black', linewidth=2)
            
            # Mostrar datos de entrenamiento si se selecciona
            if show_train:
                train_data = model_data['train']
                ax.plot(train_data.index, train_data, label='Training Data', color='gray', alpha=0.5)
            
            # Añadir predicciones para cada modelo seleccionado
            for i, model_name in enumerate(selected_models):
                model_data = results[model_name]
                color_idx = i % len(colors)
                
                # Predicciones en el conjunto de prueba
                predictions = model_data['predictions']
                ax.plot(test_data.index, predictions, label=f'{model_name} Predictions', 
                         color=colors[color_idx], linestyle='-')
                
                # Removed future predictions plotting
            
            # Añadir título y leyenda
            ax.set_title('Model Comparison')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend(loc='best')
            
            # Mostrar gráfica
            st.pyplot(fig)
            
            # Métricas detalladas para los modelos seleccionados
            st.subheader("Detailed Metrics for Selected Models")
            
            # Filtrar métricas para los modelos seleccionados
            selected_metrics = metrics_df.loc[selected_models]
            
            # Mostrar métricas en formato de tabla
            st.dataframe(selected_metrics.style.apply(highlight_min))
            
            # Visualización de métricas en gráfico de barras
            st.subheader("Metrics Visualization")
            
            # Selección de métrica a visualizar
            metric_options = metrics_df.columns.tolist()
            selected_metric = st.selectbox("Select metric to visualize", metric_options)
            
            if selected_metric:
                # Crear gráfico de barras para la métrica seleccionada
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Obtener datos para la métrica seleccionada
                metric_values = selected_metrics[selected_metric]
                
                # Crear barras con colores basados en el valor (menor es mejor)
                bars = ax.bar(metric_values.index, metric_values, color='lightblue')
                
                # Resaltar el mejor modelo
                best_model = metric_values.idxmin()
                best_idx = metric_values.index.get_loc(best_model)
                bars[best_idx].set_color('lightgreen')
                
                # Añadir etiquetas con valores
                for i, v in enumerate(metric_values):
                    ax.text(i, v + 0.01, f'{v:.4f}', ha='center')
                
                # Añadir título y etiquetas
                ax.set_title(f'{selected_metric} Comparison')
                ax.set_ylabel(selected_metric)
                ax.set_xticks(range(len(metric_values.index)))
                ax.set_xticklabels(metric_values.index, rotation=45)
                
                # Mostrar gráfico
                st.pyplot(fig)

# Export Results Page
elif page == "Export Results":
    st.header("Export Results")
    
    if 'results' not in st.session_state:
        st.warning("Please run forecasting first!")
    else:
        results = st.session_state['results']
        processed_data = st.session_state['processed_data']
        value_col = st.session_state['value_col']
        
        # Opciones de exportación
        st.subheader("Export Options")
        
        # Selección de modelos a exportar
        available_models = list(results.keys())
        selected_models = st.multiselect(
            "Select models to export",
            available_models,
            default=available_models
        )
        
        if selected_models:
            # Opciones de contenido a exportar
            export_options = st.multiselect(
                "Select what to export",
                ["Test Predictions", "Metrics", "Plots"],
                default=["Test Predictions", "Metrics"]
            )
            
            # Preparar datos para exportación
            if any(x in export_options for x in ["Test Predictions", "Future Predictions", "Metrics"]):
                # Crear un DataFrame para cada tipo de exportación
                if "Test Predictions" in export_options:
                    # Obtener datos de prueba y predicciones
                    test_data = {}
                    for model_name in selected_models:
                        model_results = results[model_name]
                        test_data[f'Actual_{value_col}'] = model_results['test']
                        test_data[f'{model_name}_Predictions'] = pd.Series(
                            model_results['predictions'], 
                            index=model_results['test'].index
                        )
                    
                    test_df = pd.DataFrame(test_data)
                    
                    # Mostrar vista previa
                    st.subheader("Test Predictions Preview")
                    st.dataframe(test_df.head())
                    
                    # Botón para descargar
                    csv = convert_df_to_csv(test_df)
                    st.download_button(
                        label="Download Test Predictions as CSV",
                        data=csv,
                        file_name="test_predictions.csv",
                        mime="text/csv"
                    )
                
                # Removed future predictions export functionality
                
                if "Metrics" in export_options:
                    # Obtener métricas
                    metrics_data = {}
                    for model_name in selected_models:
                        metrics_data[model_name] = results[model_name]['metrics']
                    
                    metrics_df = pd.DataFrame(metrics_data).T
                    
                    # Mostrar vista previa
                    st.subheader("Metrics Preview")
                    st.dataframe(metrics_df)
                    
                    # Botón para descargar
                    csv = convert_df_to_csv(metrics_df)
                    st.download_button(
                        label="Download Metrics as CSV",
                        data=csv,
                        file_name="metrics.csv",
                        mime="text/csv"
                    )
            
            # Exportar gráficos
            if "Plots" in export_options:
                st.subheader("Export Plots")
                
                # Opciones de gráficos
                plot_options = st.multiselect(
                    "Select plots to export",
                    ["Individual Model Plots", "Comparison Plot"],
                    default=["Comparison Plot"]
                )
                
                # Configuración de gráficos
                col1, col2 = st.columns(2)
                with col1:
                    show_train = st.checkbox("Include Training Data", value=False, key="export_show_train")
                with col2:
                    # Removed future predictions checkbox
                    pass
                
                # Tamaño de gráficos
                fig_width = st.slider("Figure Width (inches)", min_value=8, max_value=20, value=12)
                fig_height = st.slider("Figure Height (inches)", min_value=4, max_value=12, value=6)
                
                # Generar gráficos individuales
                if "Individual Model Plots" in plot_options:
                    for model_name in selected_models:
                        model_data = results[model_name]
                        
                        # Crear figura
                        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                        
                        # Datos de prueba y predicciones
                        test_data = model_data['test']
                        predictions = model_data['predictions']
                        
                        # Mostrar datos de entrenamiento si se selecciona
                        if show_train:
                            train_data = model_data['train']
                            ax.plot(train_data.index, train_data, label='Training Data', color='gray', alpha=0.5)
                        
                        # Datos de prueba y predicciones
                        ax.plot(test_data.index, test_data, label='Actual Values', color='blue')
                        ax.plot(test_data.index, predictions, label=f'Predictions', color='red')
                        
                        # Removed future predictions plotting
                        
                        # Añadir título y leyenda
                        ax.set_title(f'{model_name} Forecast')
                        ax.set_xlabel('Date')
                        ax.set_ylabel(value_col)
                        ax.legend(loc='best')
                        plt.tight_layout()
                        
                        # Mostrar gráfico
                        st.pyplot(fig)
                        
                        # Botón para descargar
                        buf = BytesIO()
                        fig.savefig(buf, format="png", dpi=300)
                        buf.seek(0)
                        st.download_button(
                            label=f"Download {model_name} Plot as PNG",
                            data=buf,
                            file_name=f"{model_name}_forecast.png",
                            mime="image/png"
                        )
                
                # Generar gráfico de comparación
                if "Comparison Plot" in plot_options and len(selected_models) > 1:
                    # Crear figura
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                    
                    # Colores para cada modelo
                    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
                    
                    # Datos reales (solo necesitamos mostrarlos una vez)
                    model_data = results[selected_models[0]]
                    test_data = model_data['test']
                    ax.plot(test_data.index, test_data, label='Actual Values', color='black', linewidth=2)
                    
                    # Mostrar datos de entrenamiento si se selecciona
                    if show_train:
                        train_data = model_data['train']
                        ax.plot(train_data.index, train_data, label='Training Data', color='gray', alpha=0.5)
                    
                    # Añadir predicciones para cada modelo seleccionado
                    for i, model_name in enumerate(selected_models):
                        model_data = results[model_name]
                        color_idx = i % len(colors)
                        
                        # Predicciones en el conjunto de prueba
                        predictions = model_data['predictions']
                        ax.plot(test_data.index, predictions, label=f'{model_name} Predictions', 
                                 color=colors[color_idx], linestyle='-')
                        
                        # Removed future predictions plotting
                    
                    # Añadir título y leyenda
                    ax.set_title('Model Comparison')
                    ax.set_xlabel('Date')
                    ax.set_ylabel(value_col)
                    ax.legend(loc='best')
                    plt.tight_layout()
                    
                    # Mostrar gráfico
                    st.pyplot(fig)
                    
                    # Botón para descargar
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=300)
                    buf.seek(0)
                    st.download_button(
                        label="Download Comparison Plot as PNG",
                        data=buf,
                        file_name="model_comparison.png",
                        mime="image/png"
                    )
                    
                # Exportar todas las gráficas en un solo archivo ZIP
                if len(plot_options) > 0:
                    st.subheader("Export All Selected Plots")
                    
                    # Crear un archivo ZIP con todas las gráficas
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        # Añadir gráficos individuales
                        if "Individual Model Plots" in plot_options:
                            for model_name in selected_models:
                                model_data = results[model_name]
                                
                                # Crear figura
                                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                                
                                # Datos de prueba y predicciones
                                test_data = model_data['test']
                                predictions = model_data['predictions']
                                
                                # Mostrar datos de entrenamiento si se selecciona
                                if show_train:
                                    train_data = model_data['train']
                                    ax.plot(train_data.index, train_data, label='Training Data', color='gray', alpha=0.5)
                                
                                # Datos de prueba y predicciones
                                ax.plot(test_data.index, test_data, label='Actual Values', color='blue')
                                ax.plot(test_data.index, predictions, label=f'Predictions', color='red')
                                
                                # Removed future predictions plotting
                                
                                # Añadir título y leyenda
                                ax.set_title(f'{model_name} Forecast')
                                ax.set_xlabel('Date')
                                ax.set_ylabel(value_col)
                                ax.legend(loc='best')
                                plt.tight_layout()
                                
                                # Guardar en el ZIP
                                img_buf = BytesIO()
                                fig.savefig(img_buf, format="png", dpi=300)
                                img_buf.seek(0)
                                zip_file.writestr(f"{model_name}_forecast.png", img_buf.getvalue())
                                plt.close(fig)
                        
                        # Añadir gráfico de comparación
                        if "Comparison Plot" in plot_options and len(selected_models) > 1:
                            # Crear figura
                            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                            
                            # Colores para cada modelo
                            colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
                            
                            # Datos reales (solo necesitamos mostrarlos una vez)
                            model_data = results[selected_models[0]]
                            test_data = model_data['test']
                            ax.plot(test_data.index, test_data, label='Actual Values', color='black', linewidth=2)
                            
                            # Mostrar datos de entrenamiento si se selecciona
                            if show_train:
                                train_data = model_data['train']
                                ax.plot(train_data.index, train_data, label='Training Data', color='gray', alpha=0.5)
                            
                            # Añadir predicciones para cada modelo seleccionado
                            for i, model_name in enumerate(selected_models):
                                model_data = results[model_name]
                                color_idx = i % len(colors)
                                
                                # Predicciones en el conjunto de prueba
                                predictions = model_data['predictions']
                                ax.plot(test_data.index, predictions, label=f'{model_name} Predictions', 
                                         color=colors[color_idx], linestyle='-')
                                
                                # Removed future predictions plotting
                            
                            # Añadir título y leyenda
                            ax.set_title('Model Comparison')
                            ax.set_xlabel('Date')
                            ax.set_ylabel(value_col)
                            ax.legend(loc='best')
                            plt.tight_layout()
                            
                            # Guardar en el ZIP
                            img_buf = BytesIO()
                            fig.savefig(img_buf, format="png", dpi=300)
                            img_buf.seek(0)
                            zip_file.writestr("model_comparison.png", img_buf.getvalue())
                            plt.close(fig)
                    
                    # Botón para descargar el ZIP
                    zip_buffer.seek(0)
                    st.download_button(
                        label="Download All Plots as ZIP",
                        data=zip_buffer,
                        file_name="forecast_plots.zip",
                        mime="application/zip"
                    )

# Function to train Random Forest

# Añadir una función para agregar datos de series temporales

