import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
from tqdm import tqdm


def fetch_nba_stats():
    all_data = []
    years = list(range(2015, 2025))

    for year in tqdm(years, desc="Fetching historical data"):
        try:
            url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            table = soup.find("table", {"id": "per_game_stats"})
            df = pd.read_html(str(table))[0]

            df = df[df["Player"] != "Player"].dropna(thresh=10)
            df["Season"] = f"{year - 1}-{str(year)[-2:]}"
            all_data.append(df)
        except Exception as e:
            continue

    combined_df = pd.concat(all_data)
    combined_df.to_csv("data/nba_stats.csv", index=False)
    return True