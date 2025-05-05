import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Page configuration
st.set_page_config(page_title="Tokyo Stock Market Predictor", layout="wide")
st.title("📈 Tokyo Stock Market Predictor")
st.markdown("Predict future stock prices using machine learning (Random Forest Regressor)")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter Tokyo Stock Ticker (e.g. 7203.T for Toyota):", value="7203.T")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))

# Function to load data
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["Return"] = df["Close"].pct_change()
    df = df.dropna()
    return df

# Load the data
try:
    df = load_data(ticker, start_date, end_date)
    if df.empty:
        st.warning("No data found. Please check the ticker or date range.")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Show latest data
st.subheader("📊 Historical Stock Data")
st.dataframe(df.tail())

# Features & target
features = ["Open", "High", "Low", "Volume", "SMA_10", "SMA_50", "Return"]
target = "Close"
X = df[features]
y = df[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Display performance
st.subheader("📈 Model Performance")
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Plot predictions vs actual
plot_df = pd.DataFrame({
    "Actual": y_test.values.ravel(),
    "Predicted": predictions.ravel()
}).reset_index(drop=True)

st.line_chart(plot_df)

# Download predictions
csv = plot_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Prediction CSV",
    data=csv,
    file_name="tokyo_stock_predictions.csv",
    mime="text/csv"
)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using [Streamlit](https://streamlit.io/) and [Yahoo Finance](https://finance.yahoo.com/)")
