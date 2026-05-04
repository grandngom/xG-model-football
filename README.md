# ⚽ Expected Goals (xG) Model — Football Analytics

Expected Goals (xG) model built using StatsBomb open data.

## 📊 Project Overview

This project aims to model the probability that a shot results in a goal using:

- Distance to goal
- Shot angle
- Body part
- Shot type
- First-time shot indicator

Two models are implemented:
- Baseline model (distance + angle)
- Improved model (additional features)

---

## 📈 Results

| Model    | Brier Score| Log Loss| AUC   |
|----------|------------|----------|------|
| Baseline | 0.098      | 0.332    | 0.72 |
| Improved | 0.084      | 0.296    | 0.78 |

The improved model shows better predictive performance, demonstrating the value of contextual features beyond distance and angle.

---

## 🔥 Visualization

### Heatmap of expected goals

![Heatmap](shot_heatmap_xg.png)

This heatmap shows average xG per zone on the pitch.

---

## 📂 Data

- Source: StatsBomb Open Data
- Number of matches: 50
- Total shots: 1390
- Total goals: 166

---

## 🚀 Future Work

- Add confidence intervals using bootstrap
- Incorporate defensive pressure (StatsBomb 360)
- Explore probabilistic models (mixture models, Bayesian approaches)

  ---
  
## ⚙️ How to run

```bash
python main.py
