import numpy as np

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score
)


def bootstrap_evaluation(
        y_true,
        y_pred,
        n_bootstrap=1000,
        random_state=42):

    rng = np.random.RandomState(random_state)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    brier_scores = []
    log_losses = []
    auc_scores = []

    n = len(y_true)

    for _ in range(n_bootstrap):

        indices = rng.choice(
            np.arange(n),
            size=n,
            replace=True
        )

        y_sample = y_true[indices]
        pred_sample = y_pred[indices]

        if len(np.unique(y_sample)) < 2:
            continue

        brier_scores.append(
            brier_score_loss(y_sample, pred_sample)
        )

        log_losses.append(
            log_loss(y_sample, pred_sample)
        )

        auc_scores.append(
            roc_auc_score(y_sample, pred_sample)
        )

    return {
        "brier_mean": np.mean(brier_scores),
        "brier_ci": np.percentile(
            brier_scores,
            [2.5, 97.5]
        ),

        "logloss_mean": np.mean(log_losses),
        "logloss_ci": np.percentile(
            log_losses,
            [2.5, 97.5]
        ),

        "auc_mean": np.mean(auc_scores),
        "auc_ci": np.percentile(
            auc_scores,
            [2.5, 97.5]
        )
    }