import pandas as pd
import os
from tqdm import tqdm

def fetch_nba_stats():
    all_data = []
    years = list(range(2015, 2025))
    os.makedirs("data", exist_ok=True)

    for year in tqdm(years, desc="Fetching historical data"):
        try:
            csv_url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
            df_list = pd.read_html(csv_url)
            df = df_list[0]
            df = df[df["Player"] != "Player"].dropna(thresh=10)
            df["Season"] = f"{year - 1}-{str(year)[-2:]}"
            all_data.append(df)
        except Exception as e:
            print(f"❌ Error fetching {year}: {e}")
            continue

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv("data/nba_stats.csv", index=False)
    print("✅ Data saved to data/nba_stats.csv")
    return True
