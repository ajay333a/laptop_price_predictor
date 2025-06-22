# 💻 Laptop Price Predictor

A modern machine learning application that predicts laptop prices based on various specifications using Random Forest Regressor algorithm, built with Streamlit and featuring a sleek dark theme interface.

---

## 🎯 Features

- **Real-time Price Prediction**: Get instant laptop price predictions with detailed analysis
- **Modern Dark UI**: Sleek, modern interface with custom color scheme
- **Multi-page Navigation**: Clean sidebar navigation with clickable buttons
- **Comprehensive Data Overview**: Detailed insights about the training dataset
- **Similar Laptops**: Find laptops within ₹5,000 of predicted price
- **Responsive Design**: Optimized for different screen sizes

---

## 🧠 Model Details

- **Algorithm**: Random Forest Regressor
- **Performance**: R² Score of ~0.88 (88% accuracy)
- **Features**: 18+ engineered features including CPU, GPU, RAM, Storage, Display, etc.
- **Dataset**: 1,200+ laptop records with comprehensive specifications
- **Training**: Optimized hyperparameters with cross-validation

---

## 📊 Dataset Information

### Source
The model is trained on the **"Laptop Price Prediction Dataset"** from Kaggle, which contains comprehensive laptop specifications and prices from various manufacturers.

**📎 Kaggle Dataset Link**: [Laptop Price Prediction Dataset](https://www.kaggle.com/datasets/arnabchaki/laptop-price-prediction)

### Dataset Overview
- **Total Records**: 1,300+ laptop entries
- **Price Range**: ₹20,000 - ₹2,00,000 (Indian Rupees)
- **Time Period**: Data collected from 2020-2023
- **Geographic Focus**: Indian market laptops

### Features Included
- **Hardware Specifications**: CPU, RAM, Storage, GPU
- **Display Features**: Screen size, resolution, display type
- **Physical Attributes**: Weight, laptop type
- **Software**: Operating system
- **Brand Information**: Manufacturer details

### Data Quality
- **Completeness**: 95%+ data completeness
- **Accuracy**: Verified against market prices
- **Diversity**: Multiple brands and categories
- **Currency**: All prices in Indian Rupees (₹)

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd laptop_prc_pred

# Or download and extract the project files
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Build the Model

Before running the app, you need to build and save the machine learning model:

```bash
python model_builder.py
```

This will:
- Load and clean the laptop dataset
- Extract and engineer features
- Train the Random Forest model
- Save the model and encoders as pickle files

### Step 4: Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`

---

## 🎨 App Interface

### Pages Overview

1. **🏠 Home**: Project overview and dataset statistics
2. **💰 Price Prediction**: Main prediction interface with form inputs
3. **📊 Data Info**: Information about the training dataset and source
4. **ℹ️ About**: Author information and project details

### Modern Design Features

- **Dark Theme**: Custom color scheme with primary blue (#1f77b4), dark backgrounds, and high contrast text
- **Clean Navigation**: Sidebar with clickable buttons instead of dropdown menus
- **Responsive Layout**: Four-column specification inputs for better organization
- **Visual Hierarchy**: Clear spacing and typography for better readability

---

## 📊 Features Used for Prediction

### Hardware Specifications
- **CPU**: Brand, Model, and Clock Speed
- **RAM**: Capacity in GB
- **Storage**: Type (SSD/HDD) and Capacity
- **GPU**: Brand and Model

### Display Features
- **Screen Size**: Diagonal measurement in inches
- **Resolution**: Display resolution (e.g., 1920x1080)
- **Display Type**: IPS, TN, etc.
- **Touchscreen**: Yes/No

### Physical Characteristics
- **Weight**: Laptop weight in kg
- **Laptop Type**: Ultrabook, Gaming, Notebook, etc.

### Software & Brand
- **Operating System**: Windows, macOS, Linux, etc.
- **Manufacturer Brand**: Dell, HP, Apple, etc.

---

## 🚀 How to Use

1. **Launch the App**: Run `streamlit run streamlit_app.py`
2. **Navigate**: Use the sidebar buttons to switch between pages
3. **Make Predictions**: 
   - Go to "💰 Price Prediction" page
   - Fill in the laptop specifications using the four-column layout
   - Click "🚀 Predict Price" button
   - View the predicted price and similar laptops
4. **Explore Data**: Visit "📊 Data Info" for dataset information

---

## 📁 Project Structure

```
laptop_prc_pred/
├── laptop.csv                 # Raw dataset
├── model_builder.py           # Model building and training functions
├── streamlit_app.py          # Streamlit web application with modern UI
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── random_forest_model.pkl   # Saved model (generated)
├── label_encoders.pkl        # Saved encoders (generated)
└── dummies_columns.pkl       # Saved column names (generated)
```

---

## 🔧 Technical Stack

- **Frontend**: Streamlit with custom CSS
- **Backend**: Python
- **Machine Learning**: Scikit-learn (Random Forest)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Seaborn and Matplotlib
- **Model Persistence**: Pickle

---

## 📈 Model Performance

The Random Forest model achieves excellent performance:
- **R² Score**: 0.88 (88% of variance explained)
- **Mean Squared Error**: 0.04
- **Cross-validation**: 5-fold CV used for hyperparameter tuning
- **Feature Importance**: CPU, RAM, and Storage are top predictors

---

## 🔄 How It Works

1. **Data Preprocessing**: Clean and engineer features from raw data
2. **Feature Engineering**: Extract meaningful features from text fields
3. **Model Training**: Train Random Forest with optimized hyperparameters
4. **Prediction**: Use trained model to predict prices for new specifications
5. **Similar Laptops**: Find comparable laptops within price range

---

## 📝 Example Usage

```python
from model_builder import predict_laptop_price

# Example laptop specifications
features = {
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

# Predict price
predicted_price = predict_laptop_price(features)
print(f"Predicted Price: ₹{predicted_price:,.2f}")
```

---

## 🎨 UI/UX Features

- **Modern Dark Theme**: Custom color palette for better user experience
- **Intuitive Navigation**: Clear sidebar with descriptive icons
- **Organized Inputs**: Four-column layout for laptop specifications
- **Clean Typography**: Consistent font sizes and spacing
- **Visual Feedback**: Clear prediction display with similar laptop suggestions

---

## 🤝 Contributing

This project is open for improvements and contributions. Feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🆘 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **FileNotFoundError**: Ensure `laptop.csv` is in the project directory

3. **Model not found**: Run `python model_builder.py` first to generate model files

4. **Streamlit not working**: Check if Streamlit is installed and try
   ```bash
   pip install streamlit --upgrade
   ```

### Getting Help

If you encounter any issues:
1. Check the troubleshooting section above
2. Ensure all files are in the correct directory structure
3. Verify Python version compatibility (3.8+)
4. Check that all dependencies are properly installed

---

## 🔮 Future Enhancements

Potential improvements for the project:

- Add more laptop brands and models
- Implement price trend analysis
- Add comparison features between multiple laptops
- Include user reviews and ratings
- Add export functionality for predictions
- Implement model retraining capabilities