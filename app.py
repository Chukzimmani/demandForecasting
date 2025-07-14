import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pickle
from datetime import date, datetime
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Clothing Demand Forecasting",
    page_icon="🧥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Theme Toggle Logic ---
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False

with st.sidebar:
    st.markdown("---")
    dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state['dark_mode'])
    st.session_state['dark_mode'] = dark_mode

if st.session_state['dark_mode']:
    custom_css = """
    <style>
        html, body, .main, .block-container, .main-header, .sidebar-header, .metric-card, .stInfo > div, .stSuccess > div {
            background: #181c20 !important;
            color: #fff !important;
            transition: background 0.3s, color 0.3s;
        }
        * {
            color: #fff !important;
        }
        .main-header {
            background: linear-gradient(135deg, #181c20 0%, #2c5aa0 100%);
            box-shadow: 0 4px 15px rgba(0,0,0,0.7);
        }
        .main-header h1, .main-header p, .sidebar-header, .sidebar-header h3 {
            color: #fff !important;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
        }
        .metric-card, .stInfo > div, .stSuccess > div {
            background: #23272f !important;
            color: #fff !important;
            border-left: 4px solid #4f8cff;
        }
        .stSuccess > div {
            border-left: 4px solid #22c55e;
        }
        .sidebar-header {
            background: linear-gradient(135deg, #23272f 0%, #2c5aa0 100%);
        }
        /* Style Streamlit widgets for dark mode */
        .stSlider > div, .stNumberInput > div, .stTextInput > div, .stDateInput > div, .stCheckbox > div, .stToggle > div {
            background: #23272f !important;
            color: #fff !important;
            border-radius: 8px;
            border: 1px solid #4f8cff !important;
            transition: background 0.3s, color 0.3s;
        }
        .stSlider .rc-slider-track, .stSlider .rc-slider-handle {
            background: #4f8cff !important;
            border-color: #4f8cff !important;
        }
        .stButton > button {
            background: #4f8cff !important;
            color: #fff !important;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            transition: background 0.3s, color 0.3s;
        }
        .stButton > button:hover {
            background: #2c5aa0 !important;
        }
        /* Forecast Analytics Table Styling */
        .stDataFrame, .stTable {
            background: #23272f !important;
            color: #fff !important;
            border-radius: 8px;
        }
        .stDataFrame th, .stDataFrame td, .stTable th, .stTable td {
            background: #23272f !important;
            color: #fff !important;
        }
        .stMarkdown, .stText, .stCaption, .stMetric {
            color: #fff !important;
        }
        .stMetric label {
            color: #b0c4d4 !important;
        }
        /* Chart backgrounds */
        .js-plotly-plot .plotly {
            background: #181c20 !important;
        }
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 10px;
            background: #23272f;
        }
        ::-webkit-scrollbar-thumb {
            background: #4f8cff;
            border-radius: 8px;
        }
        /* Widget focus/hover */
        .stSlider > div:focus-within, .stNumberInput > div:focus-within, .stTextInput > div:focus-within, .stDateInput > div:focus-within, .stCheckbox > div:focus-within, .stToggle > div:focus-within {
            box-shadow: 0 0 0 2px #4f8cff;
        }
    </style>
    """
else:
    custom_css = """
    <style>
        .main-header {
            background: linear-gradient(135deg, #2c5aa0 0%, #1e3a5f 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .main-header h1 {
            color: white;
            margin: 0;
            font-size: 3rem;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .main-header p {
            color: #e8f4f8;
            margin: 0.5rem 0 0 0;
            font-size: 1.2rem;
            opacity: 0.9;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .sidebar-header {
            background: linear-gradient(135deg, #2c5aa0 0%, #1e3a5f 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            color: white;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .stInfo > div {
            background-color: #f8fafc;
            border-left: 4px solid #2c5aa0;
            padding: 1rem;
            border-radius: 8px;
        }
        .stSuccess > div {
            background-color: #f0f9f4;
            border-left: 4px solid #22c55e;
            padding: 1rem;
            border-radius: 8px;
        }
    </style>
    """
st.markdown(custom_css, unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="main-header">
    <h1>🧥 Fashion Demand Analytics</h1>
    <p>AI-Powered Demand Forecasting for Modern Retail</p>
</div>
""", unsafe_allow_html=True)

# Load model with enhanced caching
@st.cache_resource
def load_model():
    with open("sarimax_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Cache forecast generation for better performance
@st.cache_data
def generate_forecast(_model, forecast_days, start_date, discount, price, competitor_price):
    # Create synthetic exogenous dataframe
    future_exog = pd.DataFrame({
        "Discount": [discount] * forecast_days,
        "Price": [price] * forecast_days,
        "Competitor Pricing": [competitor_price] * forecast_days
    })

    # Set forecast date index based on user input
    future_index = pd.date_range(start=pd.to_datetime(start_date), periods=forecast_days, freq='D')
    future_exog.index = future_index

    # Forecast using SARIMAX
    forecast = _model.get_forecast(steps=forecast_days, exog=future_exog)
    pred_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    pred_mean.index = future_index
    conf_int.index = future_index
    
    return pred_mean, conf_int, future_index

# Load the model
with st.spinner("Loading AI forecasting model..."):
    model = load_model()

# Create two columns layout
col1, col2 = st.columns([1, 2])

with col1:
    # Enhanced sidebar styling
    st.markdown("""
    <div class="sidebar-header">
        <h3>⚙️ Analytics Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Forecast Settings with improved styling
    st.markdown("### 📅 Time Settings")
    forecast_days = st.slider(
        "Forecast Horizon (days)", 
        min_value=7, 
        max_value=30, 
        value=14,
        help="Select how many days into the future you want to predict"
    )
    
    start_date = st.date_input(
        "Forecast Start Date", 
        value=date.today(),
        help="Choose the starting date for your forecast"
    )
    
    st.markdown("---")
    
    # Enhanced input section
    st.markdown("### 💰 Market Conditions")
    
    # Create input columns for better layout
    discount = st.slider(
        "Expected Discount (%)", 
        min_value=0, 
        max_value=100, 
        value=10,
        help="Average discount percentage expected during forecast period"
    )
    
    price = st.number_input(
        "Expected Price ($)", 
        min_value=0.0, 
        value=50.0, 
        step=5.0,
        help="Average product price during forecast period"
    )
    
    competitor_price = st.number_input(
        "Expected Competitor Price ($)", 
        min_value=0.0, 
        value=55.0, 
        step=5.0,
        help="Average competitor pricing during forecast period"
    )
    
    # Real-time forecast info
    st.markdown("---")
    st.info("� Forecast updates automatically as you adjust parameters!")

with col2:
    # Real-time forecast generation (removed button requirement)
    if True:  # Always generate forecast when parameters change
        
        # Generate forecast using cached function for better performance
        with st.spinner("Updating forecast..."):
            pred_mean, conf_int, future_index = generate_forecast(
                model, forecast_days, start_date, discount, price, competitor_price
            )

        # Display key metrics
        st.markdown("### 📊 Forecast Summary")
        
        # Calculate key statistics
        avg_demand = pred_mean.mean()
        total_demand = pred_mean.sum()
        max_demand = pred_mean.max()
        min_demand = pred_mean.min()
        
        # Create metrics display
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                label="📈 Avg Daily Demand",
                value=f"{avg_demand:.1f}",
                delta=f"{((avg_demand - 50) / 50 * 100):.1f}%"
            )
        
        with metric_col2:
            st.metric(
                label="📦 Total Demand",
                value=f"{total_demand:.0f}",
                delta=f"{forecast_days} days"
            )
        
        with metric_col3:
            st.metric(
                label="⬆️ Peak Demand",
                value=f"{max_demand:.1f}",
                delta=f"Day {pred_mean.idxmax().strftime('%m-%d')}"
            )
        
        with metric_col4:
            st.metric(
                label="⬇️ Min Demand",
                value=f"{min_demand:.1f}",
                delta=f"Day {pred_mean.idxmin().strftime('%m-%d')}"
            )


        # Enhanced plotting with Plotly and animation
        st.markdown("### 📈 Demand Forecast Visualization")


        # Streamlit widget for confidence interval toggle only
        show_conf = st.checkbox("Show Confidence Interval", value=True)
        dates = pred_mean.index
        forecast_vals = pred_mean.values
        conf_upper = conf_int.iloc[:, 1].values
        conf_lower = conf_int.iloc[:, 0].values

        fig = go.Figure()
        # Add forecast line (static)
        fig.add_trace(go.Scatter(
            x=dates,
            y=forecast_vals,
            mode='lines+markers',
            name='Demand Forecast',
            line=dict(color='#4f8cff' if st.session_state['dark_mode'] else '#2c5aa0', width=3),
            marker=dict(size=6, color='#4f8cff' if st.session_state['dark_mode'] else '#2c5aa0')
        ))

        # Add confidence interval if toggled
        if show_conf:
            fig.add_trace(go.Scatter(
                x=dates,
                y=conf_upper,
                mode='lines',
                line=dict(color='rgba(79, 140, 255, 0)' if st.session_state['dark_mode'] else 'rgba(44, 90, 160, 0)'),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=dates,
                y=conf_lower,
                mode='lines',
                line=dict(color='rgba(79, 140, 255, 0)' if st.session_state['dark_mode'] else 'rgba(44, 90, 160, 0)'),
                fill='tonexty',
                fillcolor='rgba(79, 140, 255, 0.18)' if st.session_state['dark_mode'] else 'rgba(44, 90, 160, 0.2)',
                name='Confidence Interval',
                hoverinfo='skip'
            ))

        # Update layout
        fig.update_layout(
            title=dict(
                text="Fashion Demand Forecast with Confidence Intervals",
                x=0.5,
                font=dict(size=20, color='#f3f6fa' if st.session_state['dark_mode'] else '#1e3a5f')
            ),
            xaxis_title="Date",
            yaxis_title="Units Sold",
            hovermode='x unified',
            template='plotly_dark' if st.session_state['dark_mode'] else 'plotly_white',
            paper_bgcolor='#181c20' if st.session_state['dark_mode'] else 'white',
            plot_bgcolor='#181c20' if st.session_state['dark_mode'] else 'white',
            height=500,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                font=dict(color='#f3f6fa' if st.session_state['dark_mode'] else '#1e3a5f')
            )
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(79, 140, 255, 0.15)' if st.session_state['dark_mode'] else 'rgba(128, 128, 128, 0.2)', color='#f3f6fa' if st.session_state['dark_mode'] else '#1e3a5f')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(79, 140, 255, 0.15)' if st.session_state['dark_mode'] else 'rgba(128, 128, 128, 0.2)', color='#f3f6fa' if st.session_state['dark_mode'] else '#1e3a5f')
        st.plotly_chart(fig, use_container_width=True)

        # Additional insights section
        st.markdown("### 🔍 Market Intelligence")

        # Calculate additional insights
        demand_trend = "increasing" if pred_mean.iloc[-1] > pred_mean.iloc[0] else "decreasing"
        price_competitiveness = "competitive" if price <= competitor_price else "premium"
        demand_volatility = pred_mean.std()
        volatility_level = "low" if demand_volatility < 10 else "moderate" if demand_volatility < 20 else "high"
        discount_impact = "strong" if discount > 15 else "moderate" if discount > 5 else "minimal"
        
        # Create insight cards with better styling
        insight_col1, insight_col2 = st.columns(2)
        
        # Calculate insight values first to avoid complex f-string formatting
        volatility_level = "low" if demand_volatility < 10 else "moderate" if demand_volatility < 20 else "high"
        discount_impact = "strong" if discount > 15 else "moderate" if discount > 5 else "minimal"
        
        with insight_col1:
            st.markdown("**💡 Key Market Insights**")
            
            # Price Strategy
            st.markdown("💰 **Price Strategy:** " + str(int(price)) + " dollar" + " is " + price_competitiveness + " vs competitor " + str(int(competitor_price)) + " dollar")
            
            # Demand Trend  
            st.markdown("📈 **Demand Trend:** " + demand_trend.title() + " pattern over " + str(forecast_days) + " days")
            
            # Market Volatility
            st.markdown("⚡ **Market Volatility:** " + f"{demand_volatility:.1f}" + " units (" + volatility_level + " variation)")
            
            # Promotion Impact
            st.markdown("🎯 **Promotion Impact:** " + str(discount) + "% discount creates " + discount_impact + " demand boost")
        
        with insight_col2:
            st.markdown("**📋 Strategic Actions**")
            
            # Inventory Target
            st.markdown("📦 **Inventory Target:** Stock " + f"{total_demand:.0f}" + " units for " + str(forecast_days) + "-day period")
            
            # Peak Preparation
            peak_date = pred_mean.idxmax().strftime('%B %d')
            st.markdown("⚠️ **Peak Preparation:** " + f"{max_demand:.1f}" + " units needed on " + peak_date)
            
            # Pricing Action
            pricing_action = "Maintain" if price_competitiveness == "competitive" else "Consider adjusting"
            st.markdown("💡 **Pricing Action:** " + pricing_action + " current strategy")
            
            # Promotion Strategy
            promo_action = "Optimize" if discount < 20 else "Monitor"
            st.markdown("🔄 **Promotion Strategy:** " + promo_action + " discount levels for ROI")
        
        # Add performance indicators
        st.markdown("### 📊 Performance Dashboard")
        
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            # Demand health indicator
            health_score = min(100, max(0, (avg_demand / 100) * 100))
            if health_score > 70:
                health_emoji = "🟢"
                health_status = "Strong"
            elif health_score > 50:
                health_emoji = "🟡"
                health_status = "Moderate"
            else:
                health_emoji = "🔴"
                health_status = "Weak"
            
            st.metric(
                label=f"{health_emoji} Demand Performance",
                value=health_status,
                delta=f"{health_score:.0f}/100 score"
            )
        
        with perf_col2:
            # Price optimization indicator
            price_diff = ((competitor_price - price) / competitor_price) * 100
            price_status = "Optimal" if abs(price_diff) < 10 else "Review"
            price_delta = f"{price_diff:+.1f}% vs market"
            
            st.metric(
                label="💰 Price Positioning",
                value=price_status,
                delta=price_delta
            )
        
        with perf_col3:
            # Market opportunity
            opportunity_score = (discount / 100) * (max_demand / avg_demand) * 100
            opportunity_status = "High" if opportunity_score > 150 else "Medium" if opportunity_score > 100 else "Low"
            
            st.metric(
                label="🎯 Market Opportunity",
                value=opportunity_status,
                delta=f"{opportunity_score:.0f} index"
            )

        # Data table with enhanced styling
        st.markdown("### 📋 Forecast Analytics Table")
        
        # Create a nice dataframe display
        forecast_df = pd.DataFrame({
            'Date': pred_mean.index.strftime('%Y-%m-%d'),
            'Predicted Demand': pred_mean.values.round(1),
            'Lower Bound': conf_int.iloc[:, 0].values.round(1),
            'Upper Bound': conf_int.iloc[:, 1].values.round(1),
            'Day of Week': pred_mean.index.strftime('%A')
        })
        
        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True
        )

        # Success message with emoji
        st.success("🎉 Analytics updated successfully! Adjust parameters above to explore different scenarios.")
    
    # Removed the else clause since we always want to show the forecast