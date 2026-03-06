# 2025 MLB Home Run Derby Forecast

A Streamlit dashboard that showcases my model’s predictions for the 2025 MLB Home Run Derby. The model uses XGBoost Poisson regression to estimate each player’s expected home runs per round. To generate tournament probabilities, the model is refit on bootstrap samples of the training data and then used to run 10,000 Monte Carlo tournament simulations, producing advancement and win probabilities for every player.

---

## Project Structure

```
derby-app/
├── app.py                     <- Streamlit dashboard
├── requirements.txt           <- Python dependencies
├── README.md                  <- This file
├── data/
│   ├── predicted_lambdas.csv  <- Predicted lambda (expected HRs) per player/round
│   ├── final_predictions.csv  <- Win, Final, and Top 4 probabilities
│   ├── MC_stats.csv           <- Monte Carlo simulation statistics
│   └── actual_results.csv     <- Actual derby results for comparison
└── assets/
    ├── hr_derby_logo.png      <- Derby logo
    └── headshots/             <- Player headshot images (PNG)

```

---

## How It Works

**Custom Historical Dataset**

The model is trained on a custom dataset of MLB Home Run Derby performances dating back to 2015. The dataset was manually assembled and curated from multiple sources, combining derby results, player statistics, and event-specific features into a structured modeling dataset.

**XGBoost Poisson Regression**

Each player's expected home runs per round (λ) are estimated using an XGBoost model with a Poisson objective. The model learns the relationship between player characteristics and derby performance to predict scoring outcomes for each round of the competition.

**Bootstrap + Monte Carlo Simulation**

To capture both model uncertainty and the randomness of the derby format:

1. Bootstrap the training data
    * 100 resampled versions of the historical dataset are generated. An XGBoost Poisson model is trained on each sample, producing 100 slightly different fitted models.

2. Generate round-level expectations
    * Each model predicts the expected number of home runs (λ) for every player and round in the 2025 derby field.

3. Simulate the tournament bracket
    * For each fitted model, 100 tournaments are simulated by drawing home run totals from a Poisson distribution and advancing players through the derby bracket.

This produces 10,000 simulated tournaments in total. Advancement and win probabilities are calculated from the aggregate results across all simulations.

---

## Dashboard Pages

**Overall Summary** -- High-level overview of the tournament forecast, including KPI cards (predicted favorite, most volatile player, highest expected HR total, and actual winner), a player ranking table, win probability bar chart, and a predicted vs. actual results comparison.

**Player Detail** -- Individual pages for each of the eight participants. Each page shows predicted expected HRs per round with bootstrap confidence intervals, Poisson distribution charts with actual results overlaid, a head-to-head comparison tool, and a short interpretation of how the model’s predictions compared to the actual derby results.

---

## Author

**Isaac Gard**



