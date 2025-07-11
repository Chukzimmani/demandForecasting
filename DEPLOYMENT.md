# Fashion Demand Analytics - Streamlit Deployment

This directory contains the deployment configuration for the Fashion Demand Analytics Streamlit app.

## Deployment Options

### 1. Streamlit Community Cloud (Recommended - Free)

**Steps to deploy:**

1. **Push your code to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Add deployment configuration for Streamlit Cloud"
   git push origin main
   ```

2. **Visit Streamlit Community Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"

3. **Configure deployment:**
   - Repository: `datasistah/clothing-demand-forecasting`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: Choose your custom URL (e.g., `fashion-demand-analytics`)

4. **Deploy:**
   - Click "Deploy!"
   - Your app will be live at: `https://[your-app-name].streamlit.app`

### 2. Alternative Free Options

#### Heroku (Free tier available)
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy to Heroku
heroku create your-app-name
git push heroku main
```

#### Railway (Free tier)
1. Connect your GitHub repo to Railway
2. Railway will auto-detect Streamlit and deploy
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

## Configuration Files

- `.streamlit/config.toml` - Streamlit theme configuration
- `runtime.txt` - Python version specification
- `requirements.txt` - Python dependencies

## Live Demo Features

Once deployed, users can:
- 🎚️ Adjust demand forecasting parameters in real-time
- 📊 Interact with dynamic Plotly visualizations
- 💡 Get AI-powered market insights and recommendations
- 📈 Monitor performance dashboard metrics
- 🔮 Perform scenario planning and "what-if" analysis

## Environment Variables (if needed)

For sensitive data, use environment variables:
```bash
# In Streamlit Cloud, add these in the app settings:
# MODEL_PATH=sarimax_model.pkl
# DATA_PATH=retail_store_inventory_preprocessed.csv
```
