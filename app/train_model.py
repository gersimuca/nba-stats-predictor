import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def train_model():
    try:
        df = pd.read_csv("data/nba_cleaned.csv")

        if "Next_PTS" not in df.columns:
            raise ValueError("Target column 'Next_PTS' missing")

        # Use data before 2023 for training, 2023 for validation
        train = df[df["Season"] < 2023]
        test = df[df["Season"] == 2023]

        X_train = train.drop(
            columns=["Player", "Next_PTS", "Season", "Team", "Pos", "Age", "FG_pct", "3P_pct", "TRB", "AST", "PTS"])
        y_train = train["Next_PTS"]
        X_test = test.drop(
            columns=["Player", "Next_PTS", "Season", "Team", "Pos", "Age", "FG_pct", "3P_pct", "TRB", "AST", "PTS"])
        y_test = test["Next_PTS"]

        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Save model
        with open("models/nba_model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Validate performance
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"Model MAE: {mae:.2f}")
        return True
    except Exception as e:
        raise RuntimeError(f"Training failed: {str(e)}")
