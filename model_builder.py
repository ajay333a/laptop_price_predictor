import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(file_path):
    """
    Load and clean the laptop dataset
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    df = df.drop(['Unnamed: 0.1'], axis=1, errors='ignore')
    df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')
    
    # Remove null values
    df = df.dropna()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Convert column names to lowercase
    df.columns = df.columns.str.lower()
    
    return df

def extract_features(df):
    """
    Extract and engineer features from the raw dataset
    """
    # CPU features
    df['cpu_brand'] = df.cpu.str.split().str[0]
    df['cpu_name'] = df.cpu.str.replace(r'\d+(?:\.\d+)?GHz', '', regex=True).str.strip()
    df['cpu_name'] = df.cpu_name.str.replace(r'^\w+', '', regex=True).str.strip()
    df['cpu_ghz'] = df.cpu.str.extract(r'(\d+(?:\.\d+)?)GHz').astype('float64')
    
    # Screen resolution features
    df['resolution'] = df['screenresolution'].str.extract(r'(\d+x\d+)')
    df['touchscreen'] = df['screenresolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
    df['display_type'] = df['screenresolution'].str.replace(r'\d+x\d+', "", regex=True).str.strip()
    df['display_type'] = df['display_type'].str.replace(r'(Full HD|Quad HD|4K Ultra HD|/|\+|Touchscreen)', '', regex=True).str.replace('/', '', regex=True).str.strip()
    
    # GPU features
    df['gpu_brand'] = df['gpu'].str.extract(r'^(\w+)')
    df['gpu_name'] = df['gpu'].str.replace(r'^(\w+)', '', regex=True).str.strip()
    
    # Memory features
    df.memory = df.memory.str.replace(r'1.0TB|1TB', "1000GB", regex=True)
    df.memory = df.memory.str.replace(r'2.0TB|2TB', "2000GB", regex=True)
    
    df['memory_list'] = df.memory.str.split('+')
    df['memory_1'] = df['memory_list'].str[0]
    df['memory_2'] = df['memory_list'].str[1]
    
    df['memory_capacity_1'] = df['memory_1'].str.extract(r'(\d+)').astype('float64')
    df['memory_type_1'] = df['memory_1'].str.replace(r'(\d+[A-Z]{2})', '', regex=True).str.strip()
    
    df['memory_capacity_2'] = df['memory_2'].str.extract(r'(\d+)').astype('float64')
    df['memory_type_2'] = df['memory_2'].str.replace(r'(\d+[A-Z]{2})', '', regex=True).str.strip()
    
    # Other numeric features
    df['ram_gb'] = df['ram'].str.replace('GB', '').astype('int')
    df['inches_size'] = pd.to_numeric(df['inches'], errors='coerce')
    df['weight_kg'] = df['weight'].replace('?', np.nan).str.replace('kg', '').astype('float64')
    
    # Remove obsolete columns
    df_clean = df.drop(columns=['ram', 'screenresolution', 'cpu', 'memory', 'memory_list',
                               'memory_1', 'memory_2', 'gpu', 'weight', 'inches'])
    
    return df_clean

def prepare_model_data(df_clean):
    """
    Prepare data for modeling with log transformation and outlier removal
    """
    # Log transform price
    df_clean['price_log'] = np.log1p(df_clean['price'])
    
    # Remove outliers
    df_clean = df_clean[df_clean['price_log'] < 12.6]
    
    # Define ordinal and nominal columns (including resolution as categorical)
    ordinal_cols = ['cpu_brand', 'gpu_brand', 'company', 'display_type', 'touchscreen', 
                   'cpu_name', 'gpu_name', 'typename', 'opsys', 'memory_type_1', 'memory_type_2', 'resolution']
    
    # Create label encoders for ordinal features
    label_encoders = {}
    for col in ordinal_cols:
        if col in df_clean.columns:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le
    
    # One-hot encode nominal features
    df_model = pd.get_dummies(df_clean, columns=ordinal_cols, drop_first=False)
    
    return df_model, label_encoders

def train_model(df_model):
    """
    Train the Random Forest model
    """
    # Prepare features and target
    X = df_model.drop(columns=['price_log', 'price'])
    y = df_model['price_log']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model with best parameters from hyperparameter tuning
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=30, max_features=15, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'R2 Score: {r2:.4f}')
    
    return rf_model, X_train.columns.tolist(), X_test, y_test, y_pred

def save_model_and_encoders(rf_model, label_encoders, dummies_columns):
    """
    Save the trained model and encoders
    """
    # Save model
    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    # Save label encoders
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Save dummy columns
    with open('dummies_columns.pkl', 'wb') as f:
        pickle.dump(dummies_columns, f)
    
    print("Model and encoders saved successfully!")

def predict_laptop_price(features):
    """
    Predict laptop price based on input features
    """
    try:
        # Load saved model and encoders
        with open('random_forest_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        with open('dummies_columns.pkl', 'rb') as f:
            dummies_columns = pickle.load(f)
        
        # Create features dataframe
        features_df = pd.DataFrame([features])
        
        # Label encode ordinal features
        ordinal_cols = ['cpu_brand', 'gpu_brand', 'company', 'display_type', 'touchscreen', 
                       'cpu_name', 'gpu_name', 'typename', 'opsys', 'memory_type_1', 'memory_type_2', 'resolution']
        
        for col in ordinal_cols:
            if col in features_df.columns:
                le = label_encoders[col]
                if features_df[col].iloc[0] not in le.classes_:
                    # Extend encoder classes if new value found
                    le.classes_ = np.append(le.classes_, features_df[col].iloc[0])
                features_df[col] = le.transform(features_df[col])
        
        # One-hot encode nominal features
        features_df = pd.get_dummies(features_df, columns=ordinal_cols, drop_first=False)
        
        # Add missing columns
        missing_cols = [col for col in dummies_columns if col not in features_df.columns]
        if missing_cols:
            missing_df = pd.DataFrame(0, index=features_df.index, columns=missing_cols)
            features_df = pd.concat([features_df, missing_df], axis=1)
        
        # Reorder columns
        features_df = features_df[dummies_columns]
        
        # Predict
        predicted_price_log = rf_model.predict(features_df)
        predicted_price = np.expm1(predicted_price_log[0])
        
        return predicted_price
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def get_unique_values(df):
    """
    Get unique values for dropdown options in the app
    """
    unique_values = {}
    
    # Get unique values for categorical features
    categorical_features = ['company', 'typename', 'opsys', 'cpu_brand', 'gpu_brand', 
                          'display_type', 'memory_type_1', 'memory_type_2', 'resolution']
    
    for feature in categorical_features:
        if feature in df.columns:
            # Convert all values to strings and handle NaN values
            unique_vals = df[feature].dropna().astype(str).unique()
            unique_values[feature] = sorted(unique_vals.tolist())
    
    return unique_values

def build_and_save_model():
    """
    Main function to build and save the model
    """
    print("Loading and cleaning data...")
    df = load_and_clean_data('laptop.csv')
    
    print("Extracting features...")
    df_clean = extract_features(df)
    
    print("Preparing model data...")
    df_model, label_encoders = prepare_model_data(df_clean)
    
    print("Training model...")
    rf_model, dummies_columns, X_test, y_test, y_pred = train_model(df_model)
    
    print("Saving model and encoders...")
    save_model_and_encoders(rf_model, label_encoders, dummies_columns)
    
    print("Model building completed successfully!")
    
    return df_clean

if __name__ == "__main__":
    # Build and save the model
    df_clean = build_and_save_model()
    
    # Test prediction
    test_features = {
        'cpu_brand': 'Intel',
        'cpu_name': 'Core i5',
        'cpu_ghz': 2.5,
        'inches_size': 15.6,
        'ram_gb': 8,
        'memory_capacity_1': 256,
        'memory_type_1': 'SSD',
        'memory_capacity_2': 0,
        'memory_type_2': 'None',
        'resolution': '1920x1080',
        'touchscreen': 0,
        'display_type': 'Full HD',
        'gpu_brand': 'Intel',
        'gpu_name': 'HD Graphics 620',
        'company': 'Dell',
        'typename': 'Notebook',
        'opsys': 'Windows 10',
        'weight_kg': 2.2
    }
    
    predicted_price = predict_laptop_price(test_features)
    if predicted_price:
        print(f"Test prediction: ₹{predicted_price:.0f}") 