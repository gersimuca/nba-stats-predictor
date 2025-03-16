import pandas as pd


def preprocess_data():
    try:
        df = pd.read_csv("data/nba_stats.csv")

        # Clean columns - REMOVE AWARDS COLUMN
        df = df.drop(columns=["Rk", "Awards"], errors="ignore")
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # Convert categorical columns
        df = pd.get_dummies(df, columns=["Team", "Pos"], prefix=["Team", "Pos"], drop_first=True)

        df.to_csv("data/nba_cleaned.csv", index=False)
        return True
    except Exception as e:
        raise RuntimeError(f"Preprocessing failed: {str(e)}")
