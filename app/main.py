import time

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

# Light Theme CSS
st.markdown("""
    <style>
        :root {
            --primary: #f8f9fa;
            --secondary: #ffffff;
            --accent: #4e73df;
            --highlight: #36b9cc;
            --text: #5a5c69;
            --border: #e3e6f0;
        }
        body { background-color: #f8f9fa; }
        .main { background-color: var(--primary); color: var(--text); }
        .stSelectbox div[data-baseweb="select"] {
            background-color: var(--secondary);
            border-radius: 8px;
            color: var(--text);
            border: 1px solid var(--border);
        }
        .metric-box {
            padding: 20px;
            border-radius: 15px;
            background: var(--secondary);
            border: 1px solid var(--border);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s;
        }
        .metric-box:hover { transform: scale(1.02); }
        .player-header {
            background: linear-gradient(45deg, #ffffff, #f0f5ff);
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border: 1px solid var(--border);
        }
        .stButton button {
            background: var(--accent) !important;
            border: none !important;
            color: white !important;
            border-radius: 8px;
            padding: 10px 24px;
            transition: all 0.3s;
        }
        .stButton button:hover {
            background: #2e59d9 !important;
            transform: translateY(-2px);
        }
        .stDataFrame { 
            background-color: var(--secondary) !important;
            border: 1px solid var(--border) !important;
        }
        .plotly-chart { background-color: rgba(0,0,0,0) !important; }
        .sidebar .sidebar-content {
            background-color: var(--secondary);
            border-right: 1px solid var(--border);
        }
        .stStatusWidget {
            border: 1px solid var(--border);
            border-radius: 10px;
            background: var(--secondary);
        }
        .stProgress > div > div {
            background-color: var(--accent) !important;
        }
        .stRadio > div {
            background-color: var(--secondary);
            border-radius: 8px;
            padding: 5px;
            border: 1px solid var(--border);
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏀 NBA Player Performance Predictor")

# Data pipeline
with st.status("🚀 Initializing application...", expanded=True) as status:
    progress_bar = st.progress(0)
    status_message = st.empty()

    try:
        # Step 1: Check for raw data
        status_message.write("🔍 Checking data requirements...")
        progress_bar.progress(5)
        time.sleep(0.3)

        if not os.path.exists("data/nba_stats.csv"):
            status_message.write("⏳ Downloading historical NBA stats...")
            # Simulate progress during download
            for i in range(5, 31):
                progress_bar.progress(i)
                time.sleep(0.1)
            fetch_nba_stats()
            status_message.write("✅ Stats downloaded!")
            progress_bar.progress(30)
        else:
            status_message.write("✅ Historical stats found")
            progress_bar.progress(30)
            time.sleep(0.3)

        # Step 2: Check for cleaned data
        status_message.write("🧹 Checking data quality...")
        progress_bar.progress(35)
        time.sleep(0.3)

        if not os.path.exists("data/nba_cleaned.csv"):
            status_message.write("🧠 Processing data...")
            # Simulate progress during processing
            for i in range(35, 56):
                progress_bar.progress(i)
                time.sleep(0.05)
            preprocess_data()
            status_message.write("✨ Data enhanced!")
            progress_bar.progress(55)
        else:
            status_message.write("✅ Clean data available")
            progress_bar.progress(55)
            time.sleep(0.3)

        # Step 3: Check for model
        status_message.write("🤖 Checking AI model...")
        progress_bar.progress(60)
        time.sleep(0.3)

        if not os.path.exists("models/nba_model.pkl"):
            status_message.write("🧠 Training prediction model...")
            # Simulate progress during training
            for i in range(60, 86):
                progress_bar.progress(i)
                time.sleep(0.05)
            train_model()
            status_message.write("🎯 Model trained!")
            progress_bar.progress(85)
        else:
            status_message.write("✅ Prediction model ready")
            progress_bar.progress(85)
            time.sleep(0.3)

        # Final loading
        status_message.write("🔍 Loading assets...")
        # Simulate final loading
        for i in range(85, 101):
            progress_bar.progress(i)
            time.sleep(0.02)

        status.update(label="✅ System ready!", state="complete", expanded=False)

    except Exception as e:
        progress_bar.progress(0)
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


# Prediction function for next season points per game
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
        prediction = round(float(raw_prediction), 1)
        pts_diff = abs(float(latest_season['Points']) - prediction)
        confidence = max(0.85 - (pts_diff * 0.05), 0.65)
        return prediction, round(confidence * 100, 1)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None


pages = [
    "Player Prediction",
    "Player Comparison",
    "Model Insights",
    "Team Overview",
    "Season Analysis",
    "Advanced Metrics",
    "Historical Performance",
    "Player Trends",
    "Shooting Analysis",
    "Playmaking Analysis",
    "Defensive Metrics",
    "Future Projections",
    "Interactive Player Filter"
]
st.sidebar.header("🔍 Navigation")
page = st.sidebar.radio("Go to", pages)

# Player Prediction Dashboard
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
                        <h3 style="color: #4e73df;">Predicted PPG</h3>
                        <h1 style="font-size: 2.5rem; margin: 1rem 0; color: #4e73df;">{prediction}</h1>
                        <div style="display: flex; align-items: center;">
                            <div style="width: 100%; height: 8px; background: #e3e6f0; border-radius: 4px;">
                                <div style="width: {confidence}%; height: 100%; background: #4e73df; border-radius: 4px;"></div>
                            </div>
                            <span style="margin-left: 1rem; color: #4e73df;">{confidence}%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Player data not available for prediction")
    with col2:
        st.markdown(f'<div class="player-header"><h2 style="color: #5a5c69;">{player_input}</h2><p>Performance Profile</p></div>',
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
                          color_discrete_sequence=["#4e73df", "#36b9cc"])  # Fixed color values
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)",
                              font_color="#5a5c69")
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

# Player Comparison Dashboard
elif page == "Player Comparison":
    st.subheader("👥 Player Comparison Tool")
    col1, col2 = st.columns(2)
    with col1:
        player1 = st.selectbox("Select Player 1:", sorted(df["Player"].unique()), index=0)
    with col2:
        player2 = st.selectbox("Select Player 2:", sorted(df["Player"].unique()), index=1)

    if player1 and player2:
        p1_stats = df[df["Player"] == player1].sort_values("Season")
        p2_stats = df[df["Player"] == player2].sort_values("Season")

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

        comparison_df = pd.concat([p1_stats, p2_stats]).sort_values("Season")
        fig = px.line(comparison_df, x="Season", y="Points", color="Player",
                      title="Points Per Game Trend")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#5a5c69"
        )
        st.plotly_chart(fig, use_container_width=True)

# Model Insights Dashboard
elif page == "Model Insights":
    st.subheader("🤖 Model Intelligence Hub")

    tabs = st.tabs(["Performance Metrics", "Feature Impact", "Prediction Accuracy"])

    with tabs[0]:
        st.markdown("### 📊 Model Performance Dashboard")
        try:
            with open("models/performance.txt", "r") as f:
                performance_data = f.readlines()
            cols = st.columns(3)
            metrics = {
                "MAE": ("Mean Absolute Error", "#4e73df"),
                "R²": ("Prediction Accuracy", "#36b9cc"),
                "Test Season": ("Evaluation Period", "#5a5c69")
            }
            for col, (key, (label, color)) in zip(cols, metrics.items()):
                value = next((line.split(": ")[1].strip() for line in performance_data if line.startswith(key)), "N/A")
                with col:
                    st.markdown(f"""
                        <div class="metric-box" style="border-left: 4px solid {color};">
                            <h4 style="color: {color};">{label}</h4>
                            <h3>{value}</h3>
                        </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.warning("Performance metrics currently unavailable")

    with tabs[1]:
        st.markdown("### 🎯 Feature Impact Analysis")
        try:
            feature_imp = pd.DataFrame({
                "Feature": model.feature_names_in_,
                "Impact": model.feature_importances_
            }).sort_values("Impact", ascending=False).head(8)
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
            fig2 = px.bar(feature_imp, x="Impact", y="Feature", orientation='h',
                          color="Impact", color_continuous_scale="Blues",
                          height=400)
            fig2.update_layout(title="Most Influential Performance Factors",
                               xaxis_title="Impact Score (0-1 scale)",
                               yaxis_title="",
                               plot_bgcolor="rgba(0,0,0,0)",
                               paper_bgcolor="rgba(0,0,0,0)",
                               font_color="#5a5c69",
                               margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig2, use_container_width=True)
            with st.expander("🔍 Feature Descriptions", expanded=True):
                st.markdown("""
                <div style="line-height: 1.6;">
                    <div style="padding: 10px; border-left: 4px solid #4e73df;">
                        <h4>3-Year Scoring Consistency</h4>
                        <p>Average points over three consecutive seasons, showing player reliability.</p>
                    </div>
                    <div style="padding: 10px; border-left: 4px solid #36b9cc; margin-top: 15px;">
                        <h4>Yearly Progress</h4>
                        <p>Percentage change in scoring compared to previous season.</p>
                    </div>
                    <div style="padding: 10px; border-left: 4px solid #5a5c69; margin-top: 15px;">
                        <h4>Scoring Efficiency</h4>
                        <p>Measures shooting effectiveness considering all scoring types.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.warning("Feature analysis currently unavailable")

    with tabs[2]:
        st.markdown("### 🔍 Prediction Accuracy")
        sample_df = df.sample(n=min(50, len(df)), random_state=42).copy()
        sample_df["Predicted"] = sample_df.apply(lambda row: model.predict(pd.DataFrame([row.drop([
            "Player", "Next Season Points", "Season", "Team", "Position", "Age",
            "Field Goal %", "3-Point %", "Rebounds", "Assists", "Points"])]))[0], axis=1)

        fig3 = px.scatter(sample_df, x="Next Season Points", y="Predicted",
                          hover_data=["Player", "Season"],
                          title="Actual vs Predicted Next Season Points")
        min_val = min(sample_df["Next Season Points"].min(), sample_df["Predicted"].min())
        max_val = max(sample_df["Next Season Points"].max(), sample_df["Predicted"].max())
        fig3.add_shape(
            type="line",
            x0=min_val, y0=min_val,
            x1=max_val, y1=max_val,
            line=dict(color="#4e73df", dash="dash")
        )
        fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)",
                           font_color="#5a5c69")
        st.plotly_chart(fig3, use_container_width=True)


# Team Overview Dashboard
elif page == "Team Overview":
    st.subheader("🏢 Team Overview Dashboard")
    team_stats = df.groupby("Team").agg({
        "Points": "mean",
        "Assists": "mean",
        "Rebounds": "mean",
        "Field Goal %": "mean",
        "3-Point %": "mean"
    }).reset_index()
    st.dataframe(team_stats)
    fig = px.bar(team_stats, x="Team", y="Points", color="Team",
                 title="Average Points per Team")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#5a5c69")
    st.plotly_chart(fig, use_container_width=True)

# Season Analysis Dashboard
elif page == "Season Analysis":
    st.subheader("📅 Season Analysis Dashboard")
    season = st.selectbox("Select Season:", sorted(df["Season"].unique()))
    season_data = df[df["Season"] == season]
    st.write(f"Statistics for season {season}")
    fig = px.box(season_data, y="Points",
                 title=f"Points Distribution in {season}")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#5a5c69")
    st.plotly_chart(fig, use_container_width=True)
    metrics = season_data.agg({
        "Points": "mean",
        "Assists": "mean",
        "Rebounds": "mean"
    })
    st.markdown("**Average Metrics**")
    st.write(metrics)

# Advanced Metrics Dashboard
elif page == "Advanced Metrics":
    st.subheader("⚡ Advanced Metrics Dashboard")
    df["Efficiency"] = (df["Points"] + df["Rebounds"] + df["Assists"]) / df["Age"]
    fig = px.scatter(df, x="Efficiency", y="Points", hover_data=["Player", "Season"],
                     title="Efficiency vs Points")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#5a5c69")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Calculated Efficiency = (Points + Rebounds + Assists) / Age")

# Historical Performance Dashboard
elif page == "Historical Performance":
    st.subheader("📜 Historical Performance Dashboard")
    player = st.selectbox("Select Player for History:", df["Player"].unique())
    player_history = df[df["Player"] == player].sort_values("Season")
    fig = px.line(player_history, x="Season", y="Points",
                  title=f"{player} Points Over Seasons")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#5a5c69")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(player_history[["Season", "Points", "Assists", "Rebounds"]])

# Player Trends Dashboard
elif page == "Player Trends":
    st.subheader("📈 Player Trends Dashboard")
    player = st.selectbox("Select Player for Trend Analysis:", df["Player"].unique())
    stat = st.selectbox("Select Statistic:", ["Points", "Assists", "Rebounds", "Field Goal %", "3-Point %"])
    player_trend = df[df["Player"] == player].sort_values("Season")
    fig = px.area(player_trend, x="Season", y=stat,
                  title=f"{player}'s {stat} Trend")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#5a5c69")
    st.plotly_chart(fig, use_container_width=True)

# Shooting Analysis Dashboard
elif page == "Shooting Analysis":
    st.subheader("🏀 Shooting Analysis Dashboard")
    fig = px.scatter(df, x="Field Goal %", y="3-Point %", hover_data=["Player", "Season"],
                     title="Field Goal % vs 3-Point %")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#5a5c69")
    st.plotly_chart(fig, use_container_width=True)

# Playmaking Analysis Dashboard
elif page == "Playmaking Analysis":
    st.subheader("🎯 Playmaking Analysis Dashboard")
    fig = px.histogram(df, x="Assists", nbins=30, title="Assists Distribution")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#5a5c69")
    st.plotly_chart(fig, use_container_width=True)

# Defensive Metrics Dashboard
elif page == "Defensive Metrics":
    st.subheader("🛡️ Defensive Metrics Dashboard")
    fig = px.violin(df, x="Rebounds", title="Rebounds Distribution")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#5a5c69")
    st.plotly_chart(fig, use_container_width=True)

# Future Projections Dashboard
elif page == "Future Projections":
    st.subheader("🔮 Future Projections Dashboard")
    player = st.selectbox("Select Player for Projection:", df["Player"].unique())
    prediction, confidence = predict_ppg(player)
    if prediction:
        st.markdown(f"""
            <div class="metric-box">
                <h3 style="color: var(--accent);">Predicted Next Season PPG</h3>
                <h1 style="font-size: 2.5rem; margin: 1rem 0; color: var(--accent);">{prediction}</h1>
                <p>Confidence: {confidence}%</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Player data not available for projection")

# Interactive Player Filter Dashboard
elif page == "Interactive Player Filter":
    st.subheader("🔍 Interactive Player Filter Dashboard")
    min_points = st.slider("Minimum Points Per Game",
                           min_value=float(df["Points"].min()),
                           max_value=float(df["Points"].max()),
                           value=float(df["Points"].min()))
    max_points = st.slider("Maximum Points Per Game",
                           min_value=float(df["Points"].min()),
                           max_value=float(df["Points"].max()),
                           value=float(df["Points"].max()))
    selected_position = st.multiselect("Select Position",
                                       options=df["Position"].unique(),
                                       default=df["Position"].unique())
    filtered_df = df[(df["Points"] >= min_points) & (df["Points"] <= max_points) &
                     (df["Position"].isin(selected_position))]
    st.dataframe(filtered_df)
    fig = px.scatter(filtered_df, x="Points", y="Assists", color="Position",
                     hover_data=["Player", "Season"],
                     title="Filtered Player Stats")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#5a5c69")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown(f"""
    <div style="text-align: center; color: #858796; padding: 1rem;">
        NBA Predictor • Data Source: basketball-reference.com • 
        Updated: {datetime.now().strftime("%Y-%m-%d")} • Version: 3.0
    </div>
""", unsafe_allow_html=True)