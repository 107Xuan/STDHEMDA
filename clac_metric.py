import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def _sanitize_scores(scores, clip_scores=True):
    """
    Replace nan/inf and optionally clip scores into [0, 1].
    NOTE: clipping is only appropriate when scores are probabilities.
    """
    scores = _to_numpy(scores).astype(float)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    if clip_scores:
        scores = np.clip(scores, 0.0, 1.0)
    return scores


def _safe_auc_aupr(y_true, y_score):
    """
    Robust AUC/AUPR computation. If y_true is single-class, return 0.0 and fallback curves.
    """
    # Defaults (diagonal ROC, simple PR)
    auc = 0.0
    aupr = 0.0
    fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    precision_list, recall_list = np.array([1.0, 0.0]), np.array([0.0, 1.0])

    try:
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
    except Exception as e:
        print(f"⚠️  Warning: AUC calculation failed: {e}")

    try:
        aupr = average_precision_score(y_true, y_score)
        precision_list, recall_list, _ = precision_recall_curve(y_true, y_score)
    except Exception as e:
        print(f"⚠️  Warning: AUPR calculation failed: {e}")

    return auc, aupr, fpr, tpr, precision_list, recall_list


def get_metric(real_labels, predict_scores, threshold=0.5, clip_scores=True):
    """
    Compute ROC/PR curve points + common binary classification metrics.

    Returns:
        tpr, fpr, recall_list, precision_list, metrics
        where metrics = (auc, aupr, accuracy, f1, recall, precision, specificity)
    """
    real_labels = _to_numpy(real_labels).astype(int)
    predict_scores = _sanitize_scores(predict_scores, clip_scores=clip_scores)

    auc, aupr, fpr, tpr, precision_list, recall_list = _safe_auc_aupr(real_labels, predict_scores)

    # Binary predictions at threshold
    predict_labels = (predict_scores >= threshold).astype(int)

    accuracy = accuracy_score(real_labels, predict_labels)
    f1 = f1_score(real_labels, predict_labels, zero_division=0)
    recall = recall_score(real_labels, predict_labels, zero_division=0)
    precision = precision_score(real_labels, predict_labels, zero_division=0)

    # Specificity (TNR)
    try:
        tn, fp, fn, tp = confusion_matrix(real_labels, predict_labels, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except Exception as e:
        print(f"⚠️  Warning: Specificity calculation failed: {e}")
        specificity = 0.0

    metrics = (auc, aupr, accuracy, f1, recall, precision, specificity)
    return tpr, fpr, recall_list, precision_list, metrics


def print_metrics(metrics, prefix=""):
    auc, aupr, accuracy, f1, recall, precision, specificity = metrics

    print(f"{prefix}Results:")
    print(f"  AUC:         {auc:.4f}")
    print(f"  AUPR:        {aupr:.4f}")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Specificity: {specificity:.4f}")


def get_metrics_dict(metrics):
    auc, aupr, accuracy, f1, recall, precision, specificity = metrics

    return {
        "AUC": auc,
        "AUPR": aupr,
        "Accuracy": accuracy,
        "F1-Score": f1,
        "Recall": recall,
        "Precision": precision,
        "Specificity": specificity,
    }


def get_metric_best_threshold(
    real_labels,
    predict_scores,
    metric="f1",
    thresholds=None,
    clip_scores=True,
):
    """
    Search best threshold on given metric.

    metric:
      - 'f1': grid search thresholds, maximize F1
      - 'youden': choose threshold that maximizes (tpr - fpr) on ROC curve

    Returns:
        metrics, best_threshold
        where metrics = (auc, aupr, accuracy, f1, recall, precision, specificity)
    """
    y_true = _to_numpy(real_labels).astype(int)
    y_score = _sanitize_scores(predict_scores, clip_scores=clip_scores)

    auc, aupr, fpr, tpr, _, _ = _safe_auc_aupr(y_true, y_score)

    if metric == "youden":
        # Use ROC thresholds directly
        try:
            fpr_curve, tpr_curve, roc_thresholds = roc_curve(y_true, y_score)
            youden = tpr_curve - fpr_curve
            best_idx = int(np.argmax(youden))
            best_threshold = float(roc_thresholds[best_idx])
            # roc_curve may return inf threshold for the first point; clamp to [0,1] if clipping enabled
            if clip_scores:
                best_threshold = float(np.clip(best_threshold, 0.0, 1.0))
        except Exception as e:
            print(f"⚠️  Warning: Youden threshold search failed: {e}")
            best_threshold = 0.5

    elif metric == "f1":
        if thresholds is None:
            thresholds = np.arange(0.05, 0.95, 0.01)

        scores = []
        for t in thresholds:
            pred = (y_score >= t).astype(int)
            scores.append(f1_score(y_true, pred, zero_division=0))

        best_idx = int(np.argmax(scores))
        best_threshold = float(thresholds[best_idx])

    else:
        raise ValueError("Unsupported metric for threshold search. Use 'f1' or 'youden'.")

    # Metrics at best threshold
    pred_labels = (y_score >= best_threshold).astype(int)
    accuracy = accuracy_score(y_true, pred_labels)
    f1 = f1_score(y_true, pred_labels, zero_division=0)
    recall = recall_score(y_true, pred_labels, zero_division=0)
    precision = precision_score(y_true, pred_labels, zero_division=0)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, pred_labels, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except Exception as e:
        print(f"⚠️  Warning: Specificity calculation failed: {e}")
        specificity = 0.0

    metrics = (auc, aupr, accuracy, f1, recall, precision, specificity)
    return metrics, best_threshold