import pandas as pd


def preprocess_data():
    try:
        df = pd.read_csv("data/nba_stats.csv")

        # Clean columns
        df = df.drop(columns=["Rk"], errors="ignore")
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # Convert team to categorical (FIXED COLUMN NAME)
        df = pd.get_dummies(df, columns=["Team"], prefix="Team", drop_first=True)

        df.to_csv("data/nba_cleaned.csv", index=False)
        return True
    except Exception as e:
        raise RuntimeError(f"Preprocessing failed: {str(e)}")
