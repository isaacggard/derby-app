# 2025 MLB Home Run Derby Forecast

A Streamlit dashboard that showcases my model's predictions of the 2025 MLB Home Run Derby. It uses an XGBoost Poisson regression model and Bootstrap Monte Carlo simulations. The model estimates each player's expected home runs per round, then simulates 10,000 tournaments to produce win probabilities and advancement odds.

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

**XGBoost Poisson Regression** -- Each player's expected HR rate (lambda) per round is estimated using an XGBoost model with a Poisson objective, trained on historical HR Derby data going back to 2015.

**Bootstrap Monte Carlo** -- To capture model uncertainty and game randomness:
1. 100 bootstrap resamples of the training data produce 100 fitted models
2. Each model predicts lambdas for every player/round
3. 100 MC tournament simulations per bootstrap model (10,000 total) use Poisson draws to simulate the bracket

The final probabilities are averages across all 10,000 simulations.

---

## Dashboard Pages

**Overall Summary** -- KPI cards (predicted favorite, most volatile, highest expected HRs, actual winner), player rankings table, win probability bar chart, and a predicted vs. actual results comparison table with Cal Raleigh highlighted as the actual winner.

**Player Detail** -- Individual pages for each of the 8 participants. Shows the player's predicted expected HRs per round with bootstrap confidence intervals, Poisson PMF distribution chart with actual result overlays, a head-to-head comparison tool, and a short interpretation of how the model's predictions compared to what actually happened.

---

## Author

**Isaac Gard**
