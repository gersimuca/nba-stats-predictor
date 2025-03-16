import requests
import pandas as pd
from bs4 import BeautifulSoup
import os


def fetch_nba_stats(years=[2020, 2021, 2022, 2023]):
    all_data = []
    for year in years:
        try:
            url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            table = soup.find("table", {"id": "per_game_stats"})

            if not table:
                raise ValueError(f"Could not find stats table for {year}")

            df = pd.read_html(str(table))[0]
            df = df[df["Player"] != "Player"]
            df = df.dropna(thresh=10).reset_index(drop=True)

            # Rename problematic columns
            df = df.rename(columns={
                'Tm': 'Team',
                'FG%': 'FG_pct',
                '3P%': '3P_pct',
                '2P%': '2P_pct',
                'eFG%': 'eFG_pct',
                'FT%': 'FT_pct'
            })

            df["Season"] = year
            all_data.append(df)
        except Exception as e:
            print(f"Error fetching data for {year}: {str(e)}")

    combined_df = pd.concat(all_data)
    combined_df.to_csv("data/nba_stats.csv", index=False)
    return True