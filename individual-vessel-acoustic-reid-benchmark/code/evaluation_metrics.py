#!/usr/bin/env python3
"""Reusable metrics for classification, verification, and retrieval interfaces."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score, roc_curve


def l2_normalize(values: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.clip(norms, eps, None)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def equal_error_rate(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def verification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "eer": equal_error_rate(y_true, scores),
    }


def cosine_scores(query_embeddings: np.ndarray, gallery_embeddings: np.ndarray) -> np.ndarray:
    query = l2_normalize(query_embeddings)
    gallery = l2_normalize(gallery_embeddings)
    return query @ gallery.T


def true_ranks(scores: np.ndarray, query_labels: np.ndarray, gallery_labels: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    query_labels = np.asarray(query_labels)
    gallery_labels = np.asarray(gallery_labels)
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked_labels = gallery_labels[order]
    matches = ranked_labels == query_labels[:, None]
    if not np.all(matches.any(axis=1)):
        raise ValueError("Every query must have at least one correct gallery identity")
    return np.argmax(matches, axis=1) + 1


def retrieval_metrics(ranks: np.ndarray, ks: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.int64)
    result = {f"recall_at_{k}": float(np.mean(ranks <= k)) for k in ks}
    result["mrr"] = float(np.mean(1.0 / ranks))
    # With one correct identity prototype per query, AP equals reciprocal rank.
    result["mAP_single_positive_gallery"] = result["mrr"]
    return result


def identity_prototypes(embeddings: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    prototypes = np.stack([embeddings[labels == label].mean(axis=0) for label in unique])
    return l2_normalize(prototypes), unique


def aggregate_recording_embeddings(
    embeddings: np.ndarray, recording_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings)
    recording_ids = np.asarray(recording_ids)
    unique = np.unique(recording_ids)
    means = np.stack([embeddings[recording_ids == recording_id].mean(axis=0) for recording_id in unique])
    return l2_normalize(means), unique


def multilabel_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predictions = np.asarray(probabilities) >= threshold
    y_true = np.asarray(y_true)
    return {
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, predictions, average="micro", zero_division=0)),
        "mAP": float(average_precision_score(y_true, probabilities, average="macro")),
        "macro_auc": float(roc_auc_score(y_true, probabilities, average="macro")),
    }

