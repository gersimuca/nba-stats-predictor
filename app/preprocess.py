import pandas as pd


def preprocess_data():
    try:
        df = pd.read_csv("data/nba_stats.csv")

        # Clean columns
        df = df.drop(columns=["Rk", "Awards"], errors="ignore")
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # Sort by player and season
        df = df.sort_values(["Player", "Season"])

        # Create next season's PTS target
        df["Next_PTS"] = df.groupby("Player")["PTS"].shift(-1)

        # Drop last season for each player (no target)
        df = df.dropna(subset=["Next_PTS"])

        # Preserve original columns for display
        display_cols = ["Team", "Pos", "Age", "FG_pct", "3P_pct", "TRB", "AST", "PTS"]
        df_display = df[["Player", "Season"] + display_cols].copy()

        # Convert categorical columns
        df = pd.get_dummies(df, columns=["Team", "Pos"], prefix=["Team", "Pos"], drop_first=True)

        # Merge back display columns
        df = pd.concat([df, df_display], axis=1)

        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        df.to_csv("data/nba_cleaned.csv", index=False)
        return True
    except Exception as e:
        raise RuntimeError(f"Preprocessing failed: {str(e)}")