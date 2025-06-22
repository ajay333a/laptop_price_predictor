import streamlit as st
import pandas as pd
import numpy as np
from model_builder import predict_laptop_price, load_and_clean_data, extract_features, get_unique_values
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for new user-defined theme
st.markdown("""
<style>
    /* --- Theme Colors --- */
    :root {
        --primary-color: #63bbea;
        --background-color: #1e1c1c;
        --secondary-background-color: #384254;
        --text-color: #dedfe8;
    }

    /* --- Main App Styling --- */
    .stApp {
        background-color: var(--background-color);
    }
    .stMarkdown, .stException, h1, h2, h3, h4, h5, h6 {
        color: var(--text-color) !important;
    }
    
    /* --- Sidebar Styling --- */
    section[data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-color) !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--primary-color);
        color: var(--primary-color) !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: var(--primary-color);
        color: var(--background-color) !important;
        border-color: var(--primary-color);
    }
    
    /* --- Custom Component Styling --- */
    .main-header {
        font-size: 3rem;
        color: var(--primary-color) !important;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: var(--text-color) !important;
        margin-bottom: 1rem;
    }
    
    /* Metric Styling */
    div[data-testid="stMetric"] {
        text-align: center;
        background-color: transparent;
        border: none;
        padding: 0;
    }
    div[data-testid="stMetric"] > label {
        color: #a0a4ac !important;
        font-weight: 400;
    }
    div[data-testid="stMetric"] > div {
        font-size: 2.25rem;
        font-weight: bold;
        color: var(--text-color);
    }
    
    .prediction-box {
        background-color: var(--secondary-background-color);
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid var(--primary-color);
        margin: 1rem 0;
    }

    /* Model Performance Cards */
    div[data-testid="stAlert"] {
        background-color: var(--secondary-background-color) !important;
        border-radius: 8px !important;
        padding: 1.5rem !important;
        border: 1px solid var(--primary-color) !important;
    }
    div[data-testid="stAlert"] p {
        color: var(--text-color) !important;
    }
    div[data-testid="stAlert"] p:first-of-type strong {
        font-size: 1.1rem;
        color: var(--primary-color) !important;
    }
    
    /* Form Elements */
    .stSelectbox div[data-baseweb="select"] > div,
    .stTextInput div[data-baseweb="input"] > input,
    .stNumberInput div[data-baseweb="input"] > input {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border-color: #4a4a4a; /* A neutral dark border */
    }
    .stSelectbox *, .stTextInput *, .stNumberInput * {
        color: var(--text-color) !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = load_and_clean_data('laptop.csv')
        df_clean = extract_features(df)
        return df_clean
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">💻 Laptop Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Predict laptop prices using machine learning with Random Forest Regressor")
    
    # Load data
    df_clean = load_data()
    
    if df_clean is None:
        st.error("Failed to load data. Please check if 'laptop.csv' file exists.")
        return
    
    # Sidebar for navigation - clickable buttons instead of dropdown
    st.sidebar.title("Navigation")
    
    # Create clickable navigation buttons
    if st.sidebar.button("🏠 Home", use_container_width=True):
        st.session_state.page = "home"
    
    if st.sidebar.button("💰 Price Prediction", use_container_width=True):
        st.session_state.page = "prediction"
    
    if st.sidebar.button("📊 Data Info", use_container_width=True):
        st.session_state.page = "data_info"
        
    if st.sidebar.button("👤 About", use_container_width=True):
        st.session_state.page = "about"
    
    # Initialize page if not set
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    # Display appropriate page
    if st.session_state.page == "home":
        show_home_page(df_clean)
    elif st.session_state.page == "prediction":
        show_prediction_page(df_clean)
    elif st.session_state.page == "data_info":
        show_data_info_page(df_clean)
    elif st.session_state.page == "about":
        show_about_page()

def show_home_page(df_clean):
    """Display the home page with project details and how it is useful"""
    st.markdown('<h2 class="sub-header">Welcome to Laptop Price Predictor</h2>', unsafe_allow_html=True)
    
    # Dataset Overview section (moved above "How it is useful")
    st.subheader("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Laptops", len(df_clean))
    
    with col2:
        avg_price = df_clean['price'].mean()
        st.metric("Average Price", f"₹{avg_price:,.0f}")
    
    with col3:
        st.metric("Unique Brands", df_clean['company'].nunique())
    
    st.markdown("---")
    
    # Specifications in four columns (not dropdown)
    st.subheader("🔧 What Specifications Can You Predict From?")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 💻 Hardware")
        st.write("• CPU Brand & Model")
        st.write("• RAM (GB)")
        st.write("• Storage Type & Capacity")
        st.write("• GPU Brand & Model")
    
    with col2:
        st.markdown("### 🖥️ Display")
        st.write("• Screen Size (inches)")
        st.write("• Resolution")
        st.write("• Display Type")
        st.write("• Touchscreen")
    
    with col3:
        st.markdown("### ⚖️ Physical")
        st.write("• Weight (kg)")
        st.write("• Laptop Type")
    
    with col4:
        st.markdown("### 🖥️ Software")
        st.write("• Operating System")
        st.write("• Brand")
    
    st.markdown("---")
    
    # Project overview and usefulness
    st.markdown("""
    ## 🎯 What is Laptop Price Predictor?
    
    The Laptop Price Predictor is an intelligent machine learning application that helps you estimate the market price of laptops based on their specifications. Whether you're a buyer looking for fair pricing or a seller setting competitive rates, this tool provides accurate price predictions using advanced AI algorithms.
    
    ## 🚀 How is it Useful?
    
    ### For Buyers:
    - **Make Informed Decisions**: Get fair price estimates before purchasing
    - **Compare Options**: Evaluate different laptop configurations
    - **Budget Planning**: Understand price ranges for desired specifications
    - **Avoid Overpaying**: Identify if a laptop is overpriced
    
    ### For Sellers:
    - **Set Competitive Prices**: Price your laptops competitively
    - **Market Analysis**: Understand current market trends
    - **Inventory Management**: Make informed pricing decisions
    
    ### For Researchers:
    - **Market Insights**: Analyze laptop pricing trends
    - **Feature Impact**: Understand which specifications affect prices most
    - **Data Analysis**: Explore comprehensive laptop market data
    """)
    
    st.markdown("---")
    
    # How to use
    st.subheader("🎯 How to Use This App")
    
    st.markdown("""
    1. **Navigate to Price Prediction**: Use the sidebar to go to the prediction page
    2. **Enter Specifications**: Fill in the laptop details you want to price
    3. **Get Instant Results**: Receive accurate price predictions with analysis
    4. **Explore Data**: Visit the About page to understand the training data
    """)
    
    # Model performance highlights
    st.subheader("🏆 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**R² Score: 0.88** \n\n This means our model explains 88% of the price variance, making it highly accurate for price predictions.")
    
    with col2:
        st.success("**Mean Squared Error: 0.037**\n\nLow error rate ensures reliable and consistent predictions across different laptop configurations.\n Note that error is for 'Log of Price' not the price in Rupees")

    st.markdown("---")
    
    # Code and Analysis section
    st.subheader("📖 Code and Analysis")
    st.markdown("""
    For a detailed walkthrough of the data analysis, feature engineering, and model building process, please refer to the complete project notebook on my data science blog.
    
    **➡️ [Read the full analysis here](https://ajay333a.quarto.pub/python_blog/posts/laptop_pc_pred/laptop_price_prediction.html)**
    """)

def show_prediction_page(df_clean):
    """Display the laptop price prediction interface"""
    st.markdown('<h2 class="sub-header">💰 Laptop Price Prediction</h2>', unsafe_allow_html=True)
    
    # Get unique values for dropdowns
    unique_values = get_unique_values(df_clean)
    
    # Create input form with consistent spacing
    with st.form("prediction_form"):
        st.subheader("Enter Laptop Specifications")
        
        # Basic Information Section
        st.markdown("### 📋 Basic Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            company = st.selectbox("Brand", unique_values.get('company', []))
        
        with col2:
            typename = st.selectbox("Laptop Type", unique_values.get('typename', []))
        
        with col3:
            opsys = st.selectbox("Operating System", unique_values.get('opsys', []))
        
        st.markdown("---")
        
        # CPU Information Section
        st.markdown("### 🔧 CPU Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cpu_brand = st.selectbox("CPU Brand", unique_values.get('cpu_brand', []))
        
        with col2:
            cpu_name = st.text_input("CPU Model (e.g., Core i5, Ryzen 7)", "Core i5")
        
        with col3:
            cpu_ghz = st.number_input("CPU Speed (GHz)", min_value=0.5, max_value=5.0, value=2.5, step=0.1)
        
        st.markdown("---")
        
        # RAM and Storage Section
        st.markdown("### 💾 Memory & Storage")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ram_gb = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 32, 64])
        
        with col2:
            memory_capacity_1 = st.number_input("Primary Storage (GB)", min_value=32, max_value=4000, value=256, step=32)
        
        with col3:
            memory_type_1 = st.selectbox("Primary Storage Type", unique_values.get('memory_type_1', []))
        
        with col4:
            memory_capacity_2 = st.number_input("Secondary Storage (GB)", min_value=0, max_value=4000, value=0, step=32)
        
        col1, col2 = st.columns(2)
        with col1:
            memory_type_2 = st.selectbox("Secondary Storage Type", ["None"] + unique_values.get('memory_type_2', []))
        
        st.markdown("---")
        
        # Display Information Section
        st.markdown("### 🖥️ Display Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            inches_size = st.number_input("Screen Size (inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
        
        with col2:
            resolution_options = ["1366x768", "1600x900", "1920x1080", "2560x1440", "3200x1800", "3840x2160"]
            resolution = st.selectbox("Screen Resolution", resolution_options)
        
        with col3:
            display_type = st.selectbox("Display Type", unique_values.get('display_type', []))
        
        with col4:
            touchscreen = st.checkbox("Touchscreen")
        
        st.markdown("---")
        
        # GPU and Physical Section
        st.markdown("### 🎮 GPU & Physical")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gpu_brand = st.selectbox("GPU Brand", unique_values.get('gpu_brand', []))
        
        with col2:
            gpu_name = st.text_input("GPU Model (e.g., HD Graphics 620, GeForce GTX 1050)", "HD Graphics 620")
        
        with col3:
            weight_kg = st.number_input("Weight (kg)", min_value=0.5, max_value=10.0, value=2.2, step=0.1)
        
        # Submit button
        submitted = st.form_submit_button("🚀 Predict Price", use_container_width=True)
    
    if submitted:
        # Prepare features dictionary
        features = {
            'cpu_brand': cpu_brand,
            'cpu_name': cpu_name,
            'cpu_ghz': cpu_ghz,
            'inches_size': inches_size,
            'ram_gb': ram_gb,
            'memory_capacity_1': memory_capacity_1,
            'memory_type_1': memory_type_1,
            'memory_capacity_2': memory_capacity_2,
            'memory_type_2': memory_type_2 if memory_type_2 != "None" else "None",
            'resolution': resolution,
            'touchscreen': 1 if touchscreen else 0,
            'display_type': display_type,
            'gpu_brand': gpu_brand,
            'gpu_name': gpu_name,
            'company': company,
            'typename': typename,
            'opsys': opsys,
            'weight_kg': weight_kg
        }
        
        # Make prediction
        with st.spinner("🤖 Making prediction..."):
            predicted_price = predict_laptop_price(features)
        
        if predicted_price:
            # Display prediction result in the box
            st.markdown(
                f'<div class="prediction-box">'
                f'<h3>💰 Predicted Price: <strong>₹{predicted_price:,.2f}</strong></h3>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Price comparison
            st.subheader("📊 Price Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Price", f"₹{predicted_price:,.0f}")
            
            with col2:
                avg_price = df_clean['price'].mean()
                st.metric("Average Market Price", f"₹{avg_price:,.0f}")
            
            # Show laptops within ₹5000 of predicted price
            st.subheader("📋 Similar Laptops (Within ₹5,000)")
            
            price_range = 5000
            similar_laptops = df_clean[
                (df_clean['price'] >= predicted_price - price_range) &
                (df_clean['price'] <= predicted_price + price_range)
            ].sort_values('price')
            
            if len(similar_laptops) > 0:
                # Display similar laptops in a table
                similar_laptops_display = similar_laptops[['company', 'typename', 'ram_gb', 'cpu_brand', 'price']].head(10)
                similar_laptops_display['price'] = similar_laptops_display['price'].apply(lambda x: f"₹{x:,.0f}")
                similar_laptops_display.columns = ['Brand', 'Type', 'RAM (GB)', 'CPU Brand', 'Price']
                
                st.dataframe(similar_laptops_display, use_container_width=True)
                st.info(f"Showing {len(similar_laptops_display)} laptops within ₹{price_range:,} of the predicted price")
            else:
                st.info("No laptops found within ₹5,000 of the predicted price.")
        
        else:
            st.error("❌ Failed to make prediction. Please check your inputs and try again.")

def show_data_info_page(df_clean):
    """Display details about the data used for training the model"""
    st.markdown('<h2 class="sub-header">📊 Data Info</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 📊 Dataset Information
    
    This Laptop Price Predictor is trained on a comprehensive dataset containing detailed specifications and prices of laptops from various manufacturers. The dataset provides a rich source of information for understanding laptop pricing patterns and market trends.
    
    **Data Source**: [Laptop Price Dataset on Kaggle](https://www.kaggle.com/datasets/muhammetvarl/laptop-price)
    """)
    
    # Dataset statistics
    st.subheader("📈 Dataset Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Basic Information")
        st.write(f"**Total Records**: {len(df_clean):,}")
        st.write(f"**Features**: {len(df_clean.columns)}")
        st.write(f"**Price Range**: ₹{df_clean['price'].min():,.0f} - ₹{df_clean['price'].max():,.0f}")
        st.write(f"**Average Price**: ₹{df_clean['price'].mean():,.0f}")
        st.write(f"**Median Price**: ₹{df_clean['price'].median():,.0f}")
    
    with col2:
        st.markdown("### Data Quality")
        st.write(f"**Missing Values**: {df_clean.isnull().sum().sum()}")
        st.write(f"**Duplicate Records**: {df_clean.duplicated().sum()}")
        st.write(f"**Unique Brands**: {df_clean['company'].nunique()}")
        st.write(f"**Unique CPU Brands**: {df_clean['cpu_brand'].nunique()}")
        st.write(f"**Unique GPU Brands**: {df_clean['gpu_brand'].nunique()}")
    
    # Model details
    st.subheader("🧠 Model Training Details")
    
    st.markdown("""
    ### Algorithm Used
    - **Random Forest Regressor**: An ensemble learning method that constructs multiple decision trees
    - **Hyperparameter Tuning**: Optimized using GridSearchCV with 5-fold cross-validation
    - **Best Parameters**: n_estimators=100, max_depth=30, max_features=15
    
    ### Feature Engineering
    The raw dataset was processed to extract meaningful features:
    - **CPU Features**: Brand, model name, and clock speed extracted from CPU strings
    - **GPU Features**: Brand and model separated from GPU descriptions
    - **Memory Features**: Storage capacity and type extracted from memory strings
    - **Display Features**: Resolution, display type, and touchscreen capability
    - **Data Cleaning**: Removed duplicates, null values, and outliers
    
    ### Model Performance
    - **R² Score**: 0.89 (89% of price variance explained)
    - **Mean Squared Error**: 0.037
    - **Cross-validation**: 5-fold CV used for robust evaluation
    """)

def show_about_page():
    """Display information about the author"""
    st.markdown('<h2 class="sub-header">👤 About the Author</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Ajay Shankar A
    
    [Twitter](https://twitter.com) | [LinkedIn](https://www.linkedin.com) | [Github](https://github.com)
    
    This is a blog for projects completed successfully by me with R-programming language. This blog will include projects from basic "Exploratory Data Analysis(EDA)" to complex "Machine Learning(ML)" projects.
    
    ---
    
    ### 🎓 Education
    - **University of Agricultural Sciences, Dharwad** | Dharwad, Karnataka
      - *Masters in Forest Biology and Tree Improvement* | Sept 2019 - Nov 2022
    - **College of Forestry, Sirsi** | Uttara Kannada, Karnataka
      - *B.Sc in Forestry* | Aug 2015 - April 2019
      
    ---
      
    ### 💼 Experience
    - **Technical Assistant** | Social Forest Department, Siruguppa | Dec 2022 - Present
    - **Research Associate** | EMPRI | May 2022 - Aug 2022
    
    ---
    
    ### 📝 Citations (Projects)
    - **Availability of Wood for Handicrafts in Karnataka** - Strengthening livelihoods and job creation.
    - **An Assessment of Wood Availability in Karnataka**

    *Information sourced from [ajay333a.quarto.pub](https://ajay333a.quarto.pub/ajay333a/about.html).*
    """)

if __name__ == "__main__":
    main() 