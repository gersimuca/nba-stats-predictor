import requests
import pandas as pd
from bs4 import BeautifulSoup
import os


def fetch_nba_stats():
    try:
        url = "https://www.basketball-reference.com/leagues/NBA_2024_per_game.html"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table", {"id": "per_game_stats"})

        if not table:
            raise ValueError("Could not find stats table on page")

        df = pd.read_html(str(table))[0]
        df = df[df["Player"] != "Player"]
        df = df.dropna(thresh=10).reset_index(drop=True)

        # Save raw data
        df.to_csv("data/nba_stats.csv", index=False)
        return True
    except Exception as e:
        raise RuntimeError(f"Scraping failed: {str(e)}")