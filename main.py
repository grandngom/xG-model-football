import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


# -------------------------------
# 1. Charger tous les fichiers JSON
# -------------------------------

data_folder = r"C:\Users\mamadou.ngom\Documents\projet xG\data"

data = []
json_files = [f for f in os.listdir(data_folder) if f.endswith(".json")]

for file_name in json_files:
    file_path = os.path.join(data_folder, file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        match_data = json.load(f)
        data.extend(match_data)


# -------------------------------
# 2. Extraire les tirs
# -------------------------------

shots = [event for event in data if event["type"]["name"] == "Shot"]


# -------------------------------
# 3. Construire le tableau des tirs
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
# 4. Créer la variable cible
# -------------------------------

df_shots["goal"] = (df_shots["outcome"] == "Goal").astype(int)


# -------------------------------
# 5. Calculer distance et angle
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
# 6. Fonction d'évaluation
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
# 7. Modèle baseline : distance + angle
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
# 8. Modèle amélioré
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
    "Modèle amélioré"
)


# -------------------------------
# 9. Comparaison avec StatsBomb
# -------------------------------

df_shots["diff_baseline_statsbomb"] = df_shots["xg_baseline"] - df_shots["statsbomb_xg"]
df_shots["diff_improved_statsbomb"] = df_shots["xg_improved"] - df_shots["statsbomb_xg"]


# -------------------------------
# 10. Exporter le dataset
# -------------------------------

df_shots.to_csv("shots_dataset.csv", index=False)


# -------------------------------
# 11. Heatmap des tirs
# -------------------------------

fig, ax = plt.subplots(figsize=(10, 7))

ax.set_xlim(80, 120)
ax.set_ylim(0, 80)

# Terrain
ax.plot([120, 120], [0, 80], color="black")
ax.plot([102, 102], [18, 62], color="black")
ax.plot([102, 120], [18, 18], color="black")
ax.plot([102, 120], [62, 62], color="black")
ax.plot([114, 114], [30, 50], color="black")
ax.plot([114, 120], [30, 30], color="black")
ax.plot([114, 120], [50, 50], color="black")

# Point du penalty et but
ax.scatter(108, 40, s=20, color="black")
ax.plot([120, 120], [36.34, 43.66], linewidth=5, color="black")

# Heatmap : couleur = xG moyen par zone
heatmap = ax.hexbin(
    df_shots["x"],
    df_shots["y"],
    C=df_shots["xg_improved"],
    reduce_C_function=np.mean,
    gridsize=18,
    mincnt=1,
    alpha=0.75
)

cbar = plt.colorbar(heatmap, ax=ax)
cbar.set_label("xG moyen amélioré")

# Buts
goals = df_shots[df_shots["goal"] == 1]

ax.scatter(
    goals["x"],
    goals["y"],
    s=70,
    marker="*",
    edgecolors="black",
    linewidths=0.6,
    label="Buts"
)

ax.set_title("Heatmap des tirs — xG moyen par zone", fontsize=14)
ax.set_xlabel("Longueur du terrain")
ax.set_ylabel("Largeur du terrain")
ax.legend(loc="upper left")
ax.set_aspect("equal")

plt.savefig("shot_heatmap_xg.png", dpi=300, bbox_inches="tight")
plt.close()


# -------------------------------
# 12. Résumé final
# -------------------------------

print("\n========== RÉSUMÉ DU PROJET xG ==========")
print("Nombre de fichiers JSON lus :", len(json_files))
print("Nombre total d'événements :", len(data))
print("Nombre total de tirs :", len(df_shots))
print("Nombre total de buts :", df_shots["goal"].sum())

print("\nRépartition des outcomes :")
print(df_shots["outcome"].value_counts())

print("\n--- Baseline : distance + angle ---")
print("Brier Score :", baseline_results["brier_score"])
print("Log Loss :", baseline_results["log_loss"])
print("AUC :", baseline_results["auc"])

print("\n--- Modèle amélioré ---")
print("Brier Score :", improved_results["brier_score"])
print("Log Loss :", improved_results["log_loss"])
print("AUC :", improved_results["auc"])

print("\nDifférence moyenne baseline - StatsBomb :")
print(df_shots["diff_baseline_statsbomb"].mean())

print("\nDifférence moyenne amélioré - StatsBomb :")
print(df_shots["diff_improved_statsbomb"].mean())

print("\nFichiers sauvegardés :")
print("- shots_dataset.csv")
print("- shot_heatmap_xg.png")