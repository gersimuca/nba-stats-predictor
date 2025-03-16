import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def train_model():
    try:
        df = pd.read_csv("data/nba_cleaned.csv")

        if "PTS" not in df.columns:
            raise ValueError("Target column 'PTS' missing")

        X = df.drop(columns=["Player", "PTS"])
        y = df["PTS"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

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