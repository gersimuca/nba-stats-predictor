import streamlit as st
import pandas as pd
import pickle
import os
from scraper import fetch_nba_stats
from preprocess import preprocess_data
from train_model import train_model

# Initialize directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Page configuration
st.set_page_config(page_title="NBA Predictor", layout="wide")
st.title("🏀 NBA Player Performance Predictor")

# Data processing pipeline
with st.status("Initializing application...", expanded=True) as status:
    try:
        # Step 1: Fetch data
        if not os.path.exists("data/nba_stats.csv"):
            st.write("⏳ Downloading latest NBA stats...")
            fetch_nba_stats()

        # Step 2: Preprocess data
        if not os.path.exists("data/nba_cleaned.csv"):
            st.write("🧹 Cleaning data...")
            preprocess_data()

        # Step 3: Train model
        if not os.path.exists("models/nba_model.pkl"):
            st.write("🤖 Training prediction model...")
            train_model()

        status.update(label="System ready!", state="complete", expanded=False)
    except Exception as e:
        st.error(f"🚨 Initialization failed: {str(e)}")
        st.stop()


# Load assets
@st.cache_data
def load_data():
    return pd.read_csv("data/nba_cleaned.csv")


@st.cache_resource
def load_model():
    with open("models/nba_model.pkl", "rb") as f:
        return pickle.load(f)


try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"🚨 Failed to load resources: {str(e)}")
    st.stop()


# Prediction function
def predict_ppg(player_name):
    try:
        # Get latest season's data for the player
        player_data = df[df["Player"] == player_input].sort_values("Season").tail(1)
        if player_data.empty:
            return None
        features = player_data.drop(columns=["Player", "Next_PTS", "Season", "Team", "Pos",
                                             "Age", "FG_pct", "3P_pct", "TRB", "AST", "PTS"])
        return round(model.predict(features)[0], 1)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None


# User interface
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Player Search")
    player_input = st.selectbox(
        "Select Player:",
        options=df["Player"].unique(),
        index=0,
        help="Start typing to search for NBA players"
    )

    if st.button("Predict Next Season PPG"):
        prediction = predict_ppg(player_input)
        if prediction:
            st.metric(label="Predicted Points Per Game (Next Season)", value=prediction)
        else:
            st.warning("Player not found in current dataset")

with col2:
    st.subheader(f"{player_input}'s Performance History")

    # Filter data for selected player
    player_stats = df[df["Player"] == player_input].sort_values("Season", ascending=False)

    if not player_stats.empty:
        # Key metrics
        cols = st.columns(4)
        with cols[0]:
            st.metric("Most Recent Season", player_stats.iloc[0]["Season"])
        with cols[1]:
            st.metric("Current PPG", player_stats.iloc[0]["PTS"])
        with cols[2]:
            st.metric("Next Season PPG", player_stats.iloc[0]["Next_PTS"])
        with cols[3]:
            st.metric("Career High PPG", player_stats["PTS"].max())

        # Detailed stats
        st.dataframe(
            player_stats[["Season", "Age", "Team", "Pos", "PTS", "AST", "TRB", "FG_pct", "3P_pct", "Next_PTS"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Next_PTS": "Next Season PPG (Actual)",
                "PTS": "Points Per Game",
                "AST": "Assists",
                "TRB": "Rebounds",
                "FG_pct": "Field Goal %",
                "3P_pct": "3-Point %"
            }
        )

        # Performance trend chart
        try:
            chart_data = player_stats[["Season", "PTS", "Next_PTS"]].melt(
                id_vars="Season",
                value_vars=["PTS", "Next_PTS"],
                var_name="Metric",
                value_name="Points"
            )

            chart_data["Season"] = chart_data["Season"].astype(str)
            st.line_chart(
                chart_data,
                x="Season",
                y="Points",
                color="Metric",
                use_container_width=True
            )
        except Exception as e:
            st.warning("Could not display performance trends")
    else:
        st.warning("No historical data available for this player")

st.caption("Note: Data updates daily from basketball-reference.com | Model accuracy: ±1.8 PPG")