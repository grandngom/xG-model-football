import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from bootstrap_utils import bootstrap_evaluation


# -------------------------------
# 1. Load all JSON files
# -------------------------------

data_folder = r"C:\Users\mamadou.ngom\Downloads\xG\data"

data = []
json_files = [f for f in os.listdir(data_folder) if f.endswith(".json")]

for file_name in json_files:
    file_path = os.path.join(data_folder, file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        match_data = json.load(f)
        data.extend(match_data)


# -------------------------------
# 2. Extract shot events
# -------------------------------

shots = [event for event in data if event["type"]["name"] == "Shot"]


# -------------------------------
# 3. Build shot dataframe
# -------------------------------

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


# -------------------------------
# 4. Create target variable
# -------------------------------

df_shots["goal"] = (df_shots["outcome"] == "Goal").astype(int)


# -------------------------------
# 5. Compute distance and angle
# -------------------------------

goal_x = 120
goal_y = 40
goal_width = 7.32

df_shots["distance"] = np.sqrt(
    (goal_x - df_shots["x"])**2 +
    (goal_y - df_shots["y"])**2
)

df_shots["angle"] = np.arctan(
    goal_width * (goal_x - df_shots["x"]) /
    ((goal_x - df_shots["x"])**2 + (goal_y - df_shots["y"])**2 - (goal_width / 2)**2)
)

df_shots["angle"] = np.abs(df_shots["angle"])


# -------------------------------
# 6. Evaluation function
# -------------------------------

def evaluate_model(y_true, y_pred, model_name):
    brier = brier_score_loss(y_true, y_pred)
    loss = log_loss(y_true, y_pred)

    if len(y_true.unique()) > 1:
        auc = roc_auc_score(y_true, y_pred)
    else:
        auc = None

    return {
        "model": model_name,
        "brier_score": brier,
        "log_loss": loss,
        "auc": auc
    }


# -------------------------------
# 7. Baseline model: distance + angle
# -------------------------------

X_baseline = df_shots[["distance", "angle"]]
y = df_shots["goal"]

stratify_option = y if y.value_counts().min() >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X_baseline,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_option
)

baseline_model = LogisticRegression()
baseline_model.fit(X_train, y_train)

df_shots["xg_baseline"] = baseline_model.predict_proba(X_baseline)[:, 1]

baseline_results = evaluate_model(
    y,
    df_shots["xg_baseline"],
    "Baseline distance + angle"
)


# -------------------------------
# 8. Improved model
# -------------------------------

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

X_train, X_test, y_train, y_test = train_test_split(
    X_improved,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_option
)

improved_model = LogisticRegression(max_iter=1000)
improved_model.fit(X_train, y_train)

df_shots["xg_improved"] = improved_model.predict_proba(X_improved)[:, 1]

improved_results = evaluate_model(
    y,
    df_shots["xg_improved"],
    "Improved model"
)


# -------------------------------
# 8 bis. Interaction-based model
# -------------------------------

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
df_interactions["distance_angle"] = df_interactions["distance"] * df_interactions["angle"]
df_interactions["distance_squared"] = df_interactions["distance"] ** 2
df_interactions["angle_squared"] = df_interactions["angle"] ** 2
df_interactions["distance_first_time"] = df_interactions["distance"] * df_interactions["first_time"]
df_interactions["angle_first_time"] = df_interactions["angle"] * df_interactions["first_time"]

# Encode categorical variables
df_interactions = pd.get_dummies(
    df_interactions,
    columns=["body_part", "shot_type"],
    drop_first=True
)

X_interactions = df_interactions.drop(columns=["goal"])
y = df_interactions["goal"]

stratify_option = y if y.value_counts().min() >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X_interactions,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_option
)

interaction_model = LogisticRegression(max_iter=2000)
interaction_model.fit(X_train, y_train)

df_shots["xg_interactions"] = interaction_model.predict_proba(X_interactions)[:, 1]

interaction_results = evaluate_model(
    y,
    df_shots["xg_interactions"],
    "Interaction-based model"
)


# -------------------------------
# 8 ter. Bootstrap validation
# -------------------------------

bootstrap_results = bootstrap_evaluation(
    y,
    df_shots["xg_interactions"],
    n_bootstrap=1000,
    random_state=42
)


# -------------------------------
# 9. Compare with StatsBomb xG
# -------------------------------

df_shots["diff_baseline_statsbomb"] = df_shots["xg_baseline"] - df_shots["statsbomb_xg"]
df_shots["diff_improved_statsbomb"] = df_shots["xg_improved"] - df_shots["statsbomb_xg"]
df_shots["diff_interactions_statsbomb"] = df_shots["xg_interactions"] - df_shots["statsbomb_xg"]


# -------------------------------
# 10. Export dataset
# -------------------------------

csv_output_path = r"C:\Users\mamadou.ngom\Documents\projet xG\shots_dataset.csv"

df_shots.to_csv(csv_output_path, index=False)


# -------------------------------
# 11. Shot heatmap
# -------------------------------

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
ax.plot([120, 120], [36.34, 43.66], linewidth=5, color="black")

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
cbar.set_label("Average xG — interaction-based model")

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

ax.set_title("Shot Heatmap — Average xG per Zone", fontsize=14)
ax.set_xlabel("Pitch Length")
ax.set_ylabel("Pitch Width")

output_path = r"C:\Users\mamadou.ngom\Documents\projet xG\shot_heatmap_xg.png"

ax.legend(loc="upper left")

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()


# -------------------------------
# 12. Final summary
# -------------------------------

print("\n========== xG PROJECT SUMMARY ==========")
print("Number of JSON files loaded:", len(json_files))
print("Total number of events:", len(data))
print("Total number of shots:", len(df_shots))
print("Total number of goals:", df_shots["goal"].sum())

print("\nOutcome distribution:")
print(df_shots["outcome"].value_counts())

print("\n--- Baseline model: distance + angle ---")
print("Brier Score:", baseline_results["brier_score"])
print("Log Loss:", baseline_results["log_loss"])
print("AUC:", baseline_results["auc"])

print("\n--- Improved model ---")
print("Brier Score:", improved_results["brier_score"])
print("Log Loss:", improved_results["log_loss"])
print("AUC:", improved_results["auc"])

print("\n--- Interaction-based model ---")
print("Brier Score:", interaction_results["brier_score"])
print("Log Loss:", interaction_results["log_loss"])
print("AUC:", interaction_results["auc"])

print("\n--- Bootstrap validation: interaction-based model ---")
print("Mean Brier Score:", bootstrap_results["brier_mean"])
print("95% CI Brier Score:", bootstrap_results["brier_ci"])

print("Mean Log Loss:", bootstrap_results["logloss_mean"])
print("95% CI Log Loss:", bootstrap_results["logloss_ci"])

print("Mean AUC:", bootstrap_results["auc_mean"])
print("95% CI AUC:", bootstrap_results["auc_ci"])

print("\nMean difference baseline - StatsBomb:")
print(df_shots["diff_baseline_statsbomb"].mean())

print("\nMean difference improved - StatsBomb:")
print(df_shots["diff_improved_statsbomb"].mean())

print("\nMean difference interactions - StatsBomb:")
print(df_shots["diff_interactions_statsbomb"].mean())

print("\nSaved files:")
print("- shots_dataset.csv")
print("- shot_heatmap_xg.png")