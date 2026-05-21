import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
    auc
)

from bootstrap_utils import bootstrap_evaluation


"""
Expected Goals (xG) Modeling Project
Author: Mamadou Ngom

This project develops and evaluates several Expected Goals (xG)
models using logistic regression, contextual features,
interaction terms, and bootstrap statistical validation.

The project includes:
- Baseline xG model
- Improved contextual model
- Interaction-based model
- Bootstrap confidence intervals
- ROC curve analysis
- Heatmap visualization
- Comparison with StatsBomb xG
"""


# --------------------------------------------------
# 1. Load all JSON files
# --------------------------------------------------

data_folder = "data"

data = []
json_files = [f for f in os.listdir(data_folder) if f.endswith(".json")]

for file_name in json_files:

    file_path = os.path.join(data_folder, file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        match_data = json.load(f)
        data.extend(match_data)


# --------------------------------------------------
# 2. Extract shot events
# --------------------------------------------------

shots = [event for event in data if event["type"]["name"] == "Shot"]


# --------------------------------------------------
# 3. Build shot dataset
# --------------------------------------------------

rows = []

for shot in shots:

    rows.append({
        "player": shot["player"]["name"],
        "team": shot["team"]["name"],
        "minute": shot["minute"],
        "second": shot["second"],
        "x": shot["location"][0],
        "y": shot["location"][1],
        "outcome": shot["shot"]["outcome"]["name"],
        "body_part": shot["shot"]["body_part"]["name"],
        "shot_type": shot["shot"]["type"]["name"],
        "first_time": shot["shot"].get("first_time", False),
        "statsbomb_xg": shot["shot"].get("statsbomb_xg", None)
    })

df_shots = pd.DataFrame(rows)


# --------------------------------------------------
# 4. Create target variable
# --------------------------------------------------

df_shots["goal"] = (df_shots["outcome"] == "Goal").astype(int)


# --------------------------------------------------
# 5. Compute distance and shooting angle
# --------------------------------------------------

goal_x = 120
goal_y = 40
goal_width = 7.32

df_shots["distance"] = np.sqrt(
    (goal_x - df_shots["x"])**2 +
    (goal_y - df_shots["y"])**2
)

df_shots["angle"] = np.arctan(
    goal_width * (goal_x - df_shots["x"]) /
    (
        (goal_x - df_shots["x"])**2 +
        (goal_y - df_shots["y"])**2 -
        (goal_width / 2)**2
    )
)

df_shots["angle"] = np.abs(df_shots["angle"])


# --------------------------------------------------
# 6. Model evaluation function
# --------------------------------------------------

def evaluate_model(y_true, y_pred, model_name):

    brier = brier_score_loss(y_true, y_pred)
    loss = log_loss(y_true, y_pred)

    if len(y_true.unique()) > 1:
        auc_score = roc_auc_score(y_true, y_pred)
    else:
        auc_score = None

    return {
        "model": model_name,
        "brier_score": brier,
        "log_loss": loss,
        "auc": auc_score
    }


# --------------------------------------------------
# 7. Baseline model: distance + angle
# --------------------------------------------------

X_baseline = df_shots[["distance", "angle"]]
y = df_shots["goal"]

stratify_option = y if y.value_counts().min() >= 2 else None

X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_baseline,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_option
)

baseline_model = LogisticRegression()

baseline_model.fit(X_train_base, y_train_base)

df_shots["xg_baseline"] = baseline_model.predict_proba(X_baseline)[:, 1]

y_pred_baseline_test = baseline_model.predict_proba(X_test_base)[:, 1]

baseline_results = evaluate_model(
    y,
    df_shots["xg_baseline"],
    "Baseline model"
)


# --------------------------------------------------
# 8. Improved contextual model
# --------------------------------------------------

df_model = df_shots[[
    "distance",
    "angle",
    "first_time",
    "body_part",
    "shot_type",
    "goal"
]].copy()

df_model["first_time"] = df_model["first_time"].astype(int)

df_model = pd.get_dummies(
    df_model,
    columns=["body_part", "shot_type"],
    drop_first=True
)

X_improved = df_model.drop(columns=["goal"])
y = df_model["goal"]

stratify_option = y if y.value_counts().min() >= 2 else None

X_train_improved, X_test_improved, y_train_improved, y_test_improved = train_test_split(
    X_improved,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_option
)

improved_model = LogisticRegression(max_iter=1000)

improved_model.fit(X_train_improved, y_train_improved)

df_shots["xg_improved"] = improved_model.predict_proba(X_improved)[:, 1]

y_pred_improved_test = improved_model.predict_proba(X_test_improved)[:, 1]

improved_results = evaluate_model(
    y,
    df_shots["xg_improved"],
    "Improved contextual model"
)


# --------------------------------------------------
# 9. Interaction-based model
# --------------------------------------------------

df_interactions = df_shots[[
    "distance",
    "angle",
    "first_time",
    "body_part",
    "shot_type",
    "goal"
]].copy()

df_interactions["first_time"] = df_interactions["first_time"].astype(int)

# Numerical interaction features
df_interactions["distance_angle"] = (
    df_interactions["distance"] *
    df_interactions["angle"]
)

df_interactions["distance_squared"] = (
    df_interactions["distance"] ** 2
)

df_interactions["angle_squared"] = (
    df_interactions["angle"] ** 2
)

df_interactions["distance_first_time"] = (
    df_interactions["distance"] *
    df_interactions["first_time"]
)

df_interactions["angle_first_time"] = (
    df_interactions["angle"] *
    df_interactions["first_time"]
)

# Encode categorical variables
df_interactions = pd.get_dummies(
    df_interactions,
    columns=["body_part", "shot_type"],
    drop_first=True
)

X_interactions = df_interactions.drop(columns=["goal"])
y = df_interactions["goal"]

stratify_option = y if y.value_counts().min() >= 2 else None

X_train_inter, X_test_inter, y_train_inter, y_test_inter = train_test_split(
    X_interactions,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_option
)

interaction_model = LogisticRegression(max_iter=2000)

interaction_model.fit(X_train_inter, y_train_inter)

df_shots["xg_interactions"] = interaction_model.predict_proba(
    X_interactions
)[:, 1]

y_pred_interactions_test = interaction_model.predict_proba(
    X_test_inter
)[:, 1]

interaction_results = evaluate_model(
    y,
    df_shots["xg_interactions"],
    "Interaction-based model"
)


# --------------------------------------------------
# 10. Bootstrap validation
# --------------------------------------------------

bootstrap_results = bootstrap_evaluation(
    y,
    df_shots["xg_interactions"],
    n_bootstrap=1000,
    random_state=42
)


# --------------------------------------------------
# 11. Comparison with StatsBomb xG
# --------------------------------------------------

df_shots["diff_baseline_statsbomb"] = (
    df_shots["xg_baseline"] -
    df_shots["statsbomb_xg"]
)

df_shots["diff_improved_statsbomb"] = (
    df_shots["xg_improved"] -
    df_shots["statsbomb_xg"]
)

df_shots["diff_interactions_statsbomb"] = (
    df_shots["xg_interactions"] -
    df_shots["statsbomb_xg"]
)


# --------------------------------------------------
# 12. Export shot dataset
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

csv_output_path = "outputs/shots_dataset.csv"

df_shots.to_csv(csv_output_path, index=False)


# --------------------------------------------------
# 13. Shot heatmap visualization
# --------------------------------------------------

os.makedirs("figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 7))

ax.set_xlim(80, 120)
ax.set_ylim(0, 80)

# Pitch lines
ax.plot([120, 120], [0, 80], color="black")

ax.plot([102, 102], [18, 62], color="black")
ax.plot([102, 120], [18, 18], color="black")
ax.plot([102, 120], [62, 62], color="black")

ax.plot([114, 114], [30, 50], color="black")
ax.plot([114, 120], [30, 30], color="black")
ax.plot([114, 120], [50, 50], color="black")

# Penalty spot and goal
ax.scatter(108, 40, s=20, color="black")

ax.plot(
    [120, 120],
    [36.34, 43.66],
    linewidth=5,
    color="black"
)

# Heatmap
heatmap = ax.hexbin(
    df_shots["x"],
    df_shots["y"],
    C=df_shots["xg_interactions"],
    reduce_C_function=np.mean,
    gridsize=18,
    mincnt=1,
    alpha=0.75
)

cbar = plt.colorbar(heatmap, ax=ax)

cbar.set_label(
    "Average xG — interaction-based model"
)

# Goals
goals = df_shots[df_shots["goal"] == 1]

ax.scatter(
    goals["x"],
    goals["y"],
    s=70,
    marker="*",
    edgecolors="black",
    linewidths=0.6,
    label="Goals"
)

ax.set_title(
    "Shot Heatmap — Average xG per Zone",
    fontsize=14
)

ax.set_xlabel("Pitch Length")
ax.set_ylabel("Pitch Width")

ax.legend(loc="upper left")

output_path = "figures/shot_heatmap_xg.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 14. ROC curve analysis
# --------------------------------------------------

fpr_base, tpr_base, _ = roc_curve(
    y_test_base,
    y_pred_baseline_test
)

roc_auc_base = auc(fpr_base, tpr_base)

fpr_improved, tpr_improved, _ = roc_curve(
    y_test_improved,
    y_pred_improved_test
)

roc_auc_improved = auc(
    fpr_improved,
    tpr_improved
)

fpr_inter, tpr_inter, _ = roc_curve(
    y_test_inter,
    y_pred_interactions_test
)

roc_auc_inter = auc(
    fpr_inter,
    tpr_inter
)

plt.rcParams.update({'font.size': 13})

plt.figure(figsize=(8, 6))

plt.plot(
    fpr_base,
    tpr_base,
    linewidth=3,
    label=f"Baseline (AUC = {roc_auc_base:.3f})"
)

plt.plot(
    fpr_improved,
    tpr_improved,
    linewidth=2,
    label=f"Improved (AUC = {roc_auc_improved:.3f})"
)

plt.plot(
    fpr_inter,
    tpr_inter,
    linewidth=2,
    label=f"Interactions (AUC = {roc_auc_inter:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curves for xG Models")

plt.legend(loc="lower right")

plt.grid(True)

roc_output_path = "figures/roc_curve_xg.png"

plt.savefig(
    roc_output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 15. Final project summary
# --------------------------------------------------

print("\n========== xG PROJECT SUMMARY ==========")

print("Number of JSON files loaded:", len(json_files))
print("Total number of events:", len(data))
print("Total number of shots:", len(df_shots))
print("Total number of goals:", df_shots["goal"].sum())

print("\nOutcome distribution:")
print(df_shots["outcome"].value_counts())

print("\n--- Baseline model ---")
print("Brier Score:", baseline_results["brier_score"])
print("Log Loss:", baseline_results["log_loss"])
print("AUC:", baseline_results["auc"])

print("\n--- Improved contextual model ---")
print("Brier Score:", improved_results["brier_score"])
print("Log Loss:", improved_results["log_loss"])
print("AUC:", improved_results["auc"])

print("\n--- Interaction-based model ---")
print("Brier Score:", interaction_results["brier_score"])
print("Log Loss:", interaction_results["log_loss"])
print("AUC:", interaction_results["auc"])

print("\n--- ROC Curve AUC values ---")
print("Baseline ROC AUC:", roc_auc_base)
print("Improved ROC AUC:", roc_auc_improved)
print("Interaction ROC AUC:", roc_auc_inter)

print("\n--- Bootstrap validation ---")

print("Mean Brier Score:")
print(bootstrap_results["brier_mean"])

print("95% Confidence Interval:")
print(bootstrap_results["brier_ci"])

print("\nMean Log Loss:")
print(bootstrap_results["logloss_mean"])

print("95% Confidence Interval:")
print(bootstrap_results["logloss_ci"])

print("\nMean AUC:")
print(bootstrap_results["auc_mean"])

print("95% Confidence Interval:")
print(bootstrap_results["auc_ci"])

print("\nAverage difference: baseline vs StatsBomb xG")
print(df_shots["diff_baseline_statsbomb"].mean())

print("\nAverage difference: improved vs StatsBomb xG")
print(df_shots["diff_improved_statsbomb"].mean())

print("\nAverage difference: interaction model vs StatsBomb xG")
print(df_shots["diff_interactions_statsbomb"].mean())

print("\nSaved files:")
print("- outputs/shots_dataset.csv")
print("- figures/shot_heatmap_xg.png")
print("- figures/roc_curve_xg.png")
