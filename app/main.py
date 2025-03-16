import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px
from datetime import datetime
from scraper import fetch_nba_stats
from preprocess import preprocess_data
from train_model import train_model

# Initialize directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="NBA Predictor",
    layout="wide",
    page_icon="🏀",
    initial_sidebar_state="expanded"
)

# Embedded Dark Theme CSS
st.markdown("""
    <style>
        :root {
            --primary: #0a192f;
            --secondary: #172a45;
            --accent: #2a4a6c;
            --highlight: #00b4d8;
            --text: #ccd6f6;
        }

        .main { background-color: var(--primary); color: var(--text); }

        .stSelectbox div[data-baseweb="select"] {
            background-color: var(--secondary);
            border-radius: 8px;
            color: var(--text);
        }

        .metric-box {
            padding: 20px;
            border-radius: 15px;
            background: var(--secondary);
            border: 1px solid #1f4068;
            transition: transform 0.2s;
        }

        .player-header {
            background: linear-gradient(45deg, var(--accent), var(--secondary));
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .stButton button {
            background: var(--accent) !important;
            border: 1px solid var(--highlight);
            color: var(--text) !important;
            border-radius: 8px;
            padding: 10px 24px;
            transition: all 0.3s;
        }

        .stDataFrame { background-color: var(--secondary) !important; }

        .plotly-chart { background-color: rgba(0,0,0,0) !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🏀 NBA Player Performance Predictor")

# Data pipeline
with st.status("🚀 Initializing application...", expanded=True) as status:
    try:
        if not os.path.exists("data/nba_stats.csv"):
            st.write("⏳ Downloading historical NBA stats...")
            fetch_nba_stats()

        if not os.path.exists("data/nba_cleaned.csv"):
            st.write("🧹 Cleaning and enhancing data...")
            preprocess_data()

        if not os.path.exists("models/nba_model.pkl"):
            st.write("🤖 Training prediction model...")
            train_model()

        status.update(label="✅ System ready!", state="complete", expanded=False)
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
        player_data = df[df["Player"] == player_name].sort_values("Season")
        if player_data.empty:
            return None, None

        latest_season = player_data.iloc[-1]
        exclude_cols = [
            "Player", "Next Season Points", "Season",
            "Team", "Position", "Age", "Field Goal %",
            "3-Point %", "Rebounds", "Assists", "Points"
        ]
        features = latest_season.drop(exclude_cols)
        raw_prediction = model.predict(pd.DataFrame([features]))[0]
        prediction = round(float(raw_prediction), 1)  # Force float conversion
        pts_diff = abs(float(latest_season['Points']) - prediction)
        confidence = max(0.85 - (pts_diff * 0.05), 0.65)
        return prediction, round(confidence * 100, 1)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None


# Main application
st.sidebar.header("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Player Prediction", "Player Comparison", "Model Insights"])

if page == "Player Prediction":
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("🔍 Player Search")
        player_input = st.selectbox(
            "Select Player:",
            options=df["Player"].unique(),
            index=0,
            help="Search for current NBA players"
        )

        if st.button("🚀 Predict Next Season PPG", use_container_width=True):
            prediction, confidence = predict_ppg(player_input)
            if prediction:
                st.markdown(f"""
                    <div class="metric-box">
                        <h3 style="color: var(--highlight);">Predicted PPG</h3>
                        <h1 style="font-size: 2.5rem; margin: 1rem 0; color: var(--highlight);">{prediction}</h1>
                        <div style="display: flex; align-items: center;">
                            <div style="width: 100%; height: 8px; background: #1f4068; border-radius: 4px;">
                                <div style="width: {confidence}%; height: 100%; background: var(--highlight); border-radius: 4px;"></div>
                            </div>
                            <span style="margin-left: 1rem; color: var(--highlight);">{confidence}%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Player data not available for prediction")

    with col2:
        st.markdown(f'<div class="player-header"><h2>{player_input}</h2><p>Performance Profile</p></div>',
                    unsafe_allow_html=True)
        player_stats = df[df["Player"] == player_input].sort_values("Season")

        if not player_stats.empty:
            cols = st.columns(4)
            metrics = [
                ("🏀 Career PPG", "Points", "mean"),
                ("📈 Peak PPG", "Points", "max"),
                ("🔄 Last Season", "Season", "last"),
                ("🎯 Accuracy", "Next Season Points", "delta")
            ]

            for col, (title, stat, calc_type) in zip(cols, metrics):
                try:
                    if calc_type == "mean":
                        value = player_stats[stat].mean()
                    elif calc_type == "max":
                        value = player_stats[stat].max()
                    elif calc_type == "last":
                        value = player_stats[stat].iloc[-1]
                    elif calc_type == "delta":
                        current = player_stats["Points"].iloc[-1]
                        predicted = player_stats["Next Season Points"].iloc[-1]
                        value = f"±{abs(round(predicted - current, 1))} PPG"

                    col.markdown(f"""
                        <div class="metric-box">
                            <h4>{title}</h4>
                            <h3>{value if isinstance(value, str) else f'{value:.1f}'}</h3>
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    col.error(f"Couldn't calculate {title}")

            fig = px.line(player_stats, x="Season", y=["Points", "Next Season Points"],
                          title="Performance Trend",
                          labels={"value": "Points Per Game", "variable": "Metric"},
                          color_discrete_sequence=["#00b4d8", "#2a4a6c"],
                          template="plotly_dark")
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white"
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📊 Detailed Statistics"):
                st.dataframe(
                    player_stats[["Season", "Age", "Team", "Position", "Points", "Assists", "Rebounds", "Field Goal %",
                                  "3-Point %"]],
                    column_config={
                        "Points": st.column_config.NumberColumn(format="%.1f"),
                        "Field Goal %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)
                    },
                    use_container_width=True
                )

elif page == "Player Comparison":
    st.subheader("👥 Player Comparison Tool")
    col1, col2 = st.columns(2)
    with col1:
        player1 = st.selectbox("Select Player 1:", df["Player"].unique(), index=0)
    with col2:
        player2 = st.selectbox("Select Player 2:", df["Player"].unique(), index=1)

    if player1 and player2:
        p1_stats = df[df["Player"] == player1]
        p2_stats = df[df["Player"] == player2]

        st.markdown("### Career Comparison")
        metrics = [
            ("Points Per Game", "Points"),
            ("Assists Per Game", "Assists"),
            ("Rebounds Per Game", "Rebounds"),
            ("Field Goal %", "Field Goal %"),
            ("3-Point %", "3-Point %"),
            ("True Shooting %", "True Shooting %")
        ]

        for i in range(0, len(metrics), 2):
            cols = st.columns(2)
            for col, (title, stat) in zip(cols, metrics[i:i + 2]):
                p1_val = p1_stats[stat].mean()
                p2_val = p2_stats[stat].mean()
                col.markdown(f"""
                    <div class="metric-box">
                        <h4>{title}</h4>
                        <div style="display: flex; justify-content: space-between;">
                            <span>{player1.split()[-1]}</span>
                            <span>{player2.split()[-1]}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <h3>{p1_val:.1f}</h3>
                            <h3>{p2_val:.1f}</h3>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        fig = px.line(pd.concat([p1_stats, p2_stats]), x="Season", y="Points", color="Player",
                      title="Points Per Game Trend",
                      template="plotly_dark")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Model Insights":
    st.subheader("🤖 Model Intelligence Hub")

    # Model Performance Dashboard
    st.markdown("### 📊 Model Performance Dashboard")
    try:
        with open("models/performance.txt", "r") as f:
            performance_data = f.readlines()

        cols = st.columns(3)
        metrics = {
            "MAE": ("Mean Absolute Error", "#00b4d8"),
            "R²": ("Prediction Accuracy", "#2a4a6c"),
            "Test Season": ("Evaluation Period", "#1f4068")
        }

        for col, (key, (label, color)) in zip(cols, metrics.items()):
            value = next((line.split(": ")[1].strip() for line in performance_data if line.startswith(key)), "N/A")
            with col:
                st.markdown(f"""
                    <div class="metric-box" style="border-color: {color};">
                        <h4 style="color: {color};">{label}</h4>
                        <h3>{value}</h3>
                    </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.warning("Performance metrics currently unavailable")

    # Feature Impact Analysis
    st.markdown("### 🎯 Feature Impact Analysis")
    try:
        feature_imp = pd.DataFrame({
            "Feature": model.feature_names_in_,
            "Impact": model.feature_importances_
        }).sort_values("Impact", ascending=False).head(8)

        # Human-readable feature names
        feature_mapping = {
            'Points_3yr_avg': '3-Year Scoring Consistency',
            'Assists_3yr_avg': 'Playmaking Consistency',
            'Rebounds_3yr_avg': 'Rebounding Consistency',
            'Points_yoy': 'Yearly Progress',
            'True Shooting %': 'Scoring Efficiency',
            'Assist/TO Ratio': 'Playmaking Quality',
            'Team_ABC': 'Team System Impact',
            'Position_XYZ': 'Positional Value'
        }

        feature_imp["Feature"] = feature_imp["Feature"].map(feature_mapping)

        # Modern visualization
        fig = px.bar(feature_imp,
                     x="Impact",
                     y="Feature",
                     orientation='h',
                     color="Impact",
                     color_continuous_scale="Tealgrn",
                     template="plotly_dark",
                     height=400)

        fig.update_layout(
            title="Most Influential Performance Factors",
            xaxis_title="Impact Score (0-1 scale)",
            yaxis_title="",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Interactive feature explorer
        with st.expander("🔍 Feature Descriptions", expanded=True):
            st.markdown("""
            <div style="line-height: 1.6;">
                <div style="padding: 10px; border-left: 4px solid #00b4d8;">
                    <h4>3-Year Scoring Consistency</h4>
                    <p>Average points over three consecutive seasons, showing player reliability</p>
                </div>
                <div style="padding: 10px; border-left: 4px solid #2a4a6c; margin-top: 15px;">
                    <h4>Yearly Progress</h4>
                    <p>Percentage change in scoring compared to previous season</p>
                </div>
                <div style="padding: 10px; border-left: 4px solid #1f4068; margin-top: 15px;">
                    <h4>Scoring Efficiency</h4>
                    <p>Measures shooting effectiveness considering all scoring types</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.warning("Feature analysis currently unavailable")

    # Advanced Data Insights
    st.markdown("### 📈 League Scoring Trends")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Player Scoring Distribution")
        st.markdown("""
        This analysis shows how scoring is distributed across all players:
        - Vertical lines show league average (blue) and top 10% threshold (teal)
        - Helps identify scoring trends and outliers
        """)

        fig = px.histogram(df,
                           x="Points",
                           nbins=25,
                           labels={"Points": "Points Per Game"},
                           template="plotly_dark",
                           color_discrete_sequence=["#00b4d8"])

        # Add reference lines
        avg_ppg = df["Points"].mean()
        top_10 = df["Points"].quantile(0.9)
        fig.add_vline(x=avg_ppg, line_color="#2a4a6c", annotation_text="League Average")
        fig.add_vline(x=top_10, line_color="#1f4068", annotation_text="Top 10% Threshold")

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            bargap=0.08,
            xaxis_title="Points Per Game",
            yaxis_title="Number of Players",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Scoring Benchmarks")
        st.markdown(f"""
        <div class="metric-box" style="margin-bottom: 20px;">
            <h4>🏆 Elite Scorers</h4>
            <h3>{len(df[df['Points'] >= 25])}</h3>
            <p>Players averaging 25+ PPG</p>
        </div>
        <div class="metric-box" style="margin-bottom: 20px;">
            <h4>⭐ All-Star Level</h4>
            <h3>{len(df[df['Points'] >= 20])}</h3>
            <p>Players averaging 20+ PPG</p>
        </div>
        <div class="metric-box">
            <h4>📊 League Average</h4>
            <h3>{df['Points'].mean():.1f}</h3>
            <p>Points Per Game</p>
        </div>
        """, unsafe_allow_html=True)

    # Data Composition
    st.markdown("### 🧩 Dataset Overview")
    cols = st.columns(4)
    metrics = [
        ("👥 Players", df["Player"].nunique(), "#00b4d8"),
        ("📅 Seasons", f"{df['Season'].min()} - {df['Season'].max()}", "#2a4a6c"),
        ("📈 Avg Age", f"{df['Age'].mean():.1f}", "#1f4068"),
        ("🏀 Records", f"{len(df):,}", "#00b4d8")
    ]

    for col, (label, value, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
                <div class="metric-box" style="border-color: {color};">
                    <h4 style="color: {color};">{label}</h4>
                    <h3>{value}</h3>
                </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 1rem;">
        NBA Predictor • Data Source: basketball-reference.com • 
        Updated: {datetime.now().strftime("%Y-%m-%d")} • Version: 2.2
    </div>
""", unsafe_allow_html=True)
