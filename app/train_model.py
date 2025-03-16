import pandas as pd
import pickle
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_model():
    try:
        df = pd.read_csv("data/nba_cleaned.csv")

        # Temporal validation split
        seasons = sorted(df["Season"].unique())
        test_season = seasons[-1]
        train = df[df["Season"] != test_season]
        test = df[df["Season"] == test_season]

        # Feature selection
        exclude_cols = [
            "Player", "Next Season Points", "Season",
            "Team", "Position", "Age", "Field Goal %",
            "3-Point %", "Rebounds", "Assists", "Points"
        ]
        X_train = train.drop(columns=exclude_cols)
        y_train = train["Next Season Points"]
        X_test = test.drop(columns=exclude_cols)
        y_test = test["Next Season Points"]

        # Model configuration
        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=20,
            eval_metric='mae'
        )

        # Train model
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Evaluate
        test_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, test_pred)
        r2 = r2_score(y_test, test_pred)

        # Save artifacts
        with open("models/nba_model.pkl", "wb") as f:
            pickle.dump(model, f)

        with open("models/performance.txt", "w") as f:
            f.write(f"Model Evaluation ({test_season} season)\n")
            f.write(f"MAE: {mae:.2f}\n")  # Force 2 decimal places
            f.write(f"R²: {r2:.2f}\n")  # Force 2 decimal places
            f.write(f"Training Seasons: {len(train['Season'].unique())}\n")
            f.write(f"Test Season: {test_season}\n")
            f.write(f"Model Type: XGBoost Regressor\n")
            f.write(f"Features Used: {len(X_train.columns)}\n")

        return True
    except Exception as e:
        raise RuntimeError(f"Training failed: {str(e)}")