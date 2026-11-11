# Zomato Bangalore Restaurants: What Is Linked to a High Rating?

An exploratory analysis and classification model on Zomato restaurant data from Bangalore.

## Overview
This project looks at which restaurant features (cost, table booking, online ordering, restaurant type, location and popularity) are associated with higher ratings, and how well those features can predict a highly rated restaurant (rating of 4.0 or above).

**Dataset:** Zomato Bangalore Restaurants (Kaggle), 51,717 rows and 17 columns.

## What I did
- **Cleaning:** The same restaurant appeared many times (once per delivery zone), so I removed duplicates by name and address (12,499 restaurants left). I then dropped restaurants with no rating or cost, leaving 9,284.
- **Distributions:** `votes` is highly skewed (mean 242 vs median 53), so I applied a log transform instead of deleting outliers.
- **Analysis:** correlation (Pearson and Spearman), ratings by location, table booking and online ordering, and a check on whether cost or restaurant type explains the differences.
- **Hypothesis tests:** Mann-Whitney U tests for online ordering and table booking.
- **Model:** Random Forest classifier for highly rated restaurants, with a baseline, permutation importance and 5-fold cross-validation.

## Key findings
- **Popularity is most strongly linked to rating:** votes and rating have a Spearman correlation of 0.63, while cost and rating have 0.28.
- **Location:** Indiranagar (3.78) and Jayanagar (3.76) have the highest average ratings among the top 10 areas, but the whole range is only about 0.3 points.
- **Table booking:** restaurants offering it are rated higher (4.11 vs 3.57), but they also cost about three times more (median ₹1,200 vs ₹400). Within the same cost band the gap shrinks to about 0.3-0.4 points but does not disappear, and it also holds inside Casual Dining and Cafes.
- **Online ordering:** the gap is small (median 3.7 vs 3.6) but consistent, about 0.1 points in every cost band. Both Mann-Whitney tests are significant, but significance is not the same as a large effect.

## Model results
Highly rated restaurants are 22.7% of the data, so I used a baseline that always predicts "not highly rated" (accuracy 0.773) and focused on F1 for the high-rated class.

| Model | F1 (high-rated class) | Recall | Precision |
|---|---|---|---|
| Random Forest with votes | 0.67 (CV mean 0.68) | 0.80 | 0.57 |
| Random Forest without votes | 0.55 (CV mean 0.55) | 0.62 | 0.48 |

Most of the predictive power comes from `votes`. Without it, cost, restaurant type, table booking and online ordering explain only part of a restaurant's rating.

## Limitations
- These are associations, not causes. Food quality, service and ambience are not in the data.
- `votes` is collected after a restaurant becomes popular, so it partly reflects the rating itself.
- Only 6 budget restaurants offer table booking, so that group was ignored in the conclusions.
- The model uses a small set of features and no hyperparameter tuning.

## Tools
Python, pandas, NumPy, matplotlib, seaborn, SciPy, scikit-learn.

## How to run
1. Download the dataset from Kaggle (Zomato Bangalore Restaurants).
2. Open `zomato-bangalore-analysis.ipynb` and update the file path in the data loading cell.
3. Run all cells.

**Author:** AKANKSHA KURIAL
