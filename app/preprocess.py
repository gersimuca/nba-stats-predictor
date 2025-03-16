import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def preprocess_data():
    try:
        df = pd.read_csv("data/nba_stats.csv")

        # Clean and standardize data
        df['Player'] = df['Player'].str.strip().str.title()
        df = df.drop(columns=["Rk", "Awards", "Unnamed: 0"], errors="ignore")
        df = df[df["Player"].notna()].reset_index(drop=True)

        # Rename columns for clarity
        df = df.rename(columns={
            'Tm': 'Team',
            'Pos': 'Position',
            'PTS': 'Points',
            'AST': 'Assists',
            'TRB': 'Rebounds',
            'FG%': 'Field Goal %',
            '3P%': '3-Point %',
            'MP': 'Minutes Played',
            'Age': 'Age'
        })

        # Convert numeric types
        numeric_cols = ['Age', 'Field Goal %', '3-Point %', 'Rebounds', 'Assists', 'Points', 'Minutes Played']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Feature engineering
        df = df.sort_values(["Player", "Season"])
        for stat in ['Points', 'Assists', 'Rebounds']:
            df[f'{stat}_3yr_avg'] = df.groupby("Player")[stat].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
            df[f'{stat}_yoy'] = df.groupby("Player")[stat].pct_change().replace([np.inf, -np.inf], np.nan)

        # Fixed True Shooting % calculation
        df['True Shooting %'] = df['Points'] / (2 * (df['FGA'] + 0.44 * df['FTA'].replace(0, np.nan) + 1e-6))
        df['Assist/TO Ratio'] = (df['Assists'] / df['TOV'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

        # Handle missing/invalid values
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Target variable
        df["Next Season Points"] = df.groupby("Player")["Points"].shift(-1)
        df = df.dropna(subset=["Next Season Points"])

        # Encode categoricals
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_features = encoder.fit_transform(df[["Team", "Position"]])
        encoded_df = pd.DataFrame(
            encoded_features,
            columns=encoder.get_feature_names_out(["Team", "Position"])
        )

        final_df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        final_df.to_csv("data/nba_cleaned.csv", index=False)
        return True

    except Exception as e:
        raise RuntimeError(f"Preprocessing failed: {str(e)}")