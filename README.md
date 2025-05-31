# NBA Stats Predictor

## Overview
The **NBA Stats Predictor** is a machine learning application that predicts a player's **points per game (PPG)** based on historical performance data. This project utilizes **web scraping, data preprocessing, and machine learning** to provide insights into player performance.

## Features
- **Data Scraping:** Extracts NBA player stats from [Basketball Reference](https://www.basketball-reference.com/).
- **Data Processing:** Cleans and formats the raw data for accurate predictions.
- **Machine Learning Model:** Trains a **RandomForestRegressor** to predict PPG.
- **Performance Prediction:** Estimates future player performance based on historical data.
- **Web-Based Interface:** Uses **Streamlit** for an interactive user experience.

## Installation
Follow these steps to set up and run the NBA Stats Predictor:

1. **Clone this repository:**
   ```sh
   git clone https://github.com/your-repo/nba-stats-predictor.git
   cd nba-stats-predictor
   ```
2. **Set up a virtual environment (recommended):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the Streamlit app:**
   ```sh
   streamlit run app/main.py
   ```

## Usage
1. **Enter a player's name** in the input field.
2. Click the **Predict** button.
3. View the **predicted PPG** based on historical stats.

## 📂 Project Structure
```
📦 nba-stats-predictor
├── 📂 app
│   ├── main.py            # Streamlit web app
│   ├── preprocess.py      # Web scraping & data cleaning
│   ├── scraper.py         # Preprocessing & feature engineering
│   ├── train_model.py     # Training the ML model
├── 📂 data
│   ├── nba_stats.csv      # Raw scraped data
│   ├── nba_cleaned.csv    # Processed dataset
├── 📂 models
│   ├── nba_model.pkl      # Trained machine learning model
│   ├── performance.txt    # Model performance metrics
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

## How It Works
1. **Scrape Data:** Fetches NBA player statistics using **BeautifulSoup**.
2. **Preprocess Data:** Cleans the dataset and converts categorical values.
3. **Train Model:** Trains a **RandomForestRegressor** on historical data.
4. **Make Predictions:** Uses the trained model to predict PPG for a given player.

## Contributors
- **Gersi Muca** – [GitHub](https://github.com/gersimuca)

## License
This project is licensed under the **MIT License**.

