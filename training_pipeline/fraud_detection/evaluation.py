import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    ConfusionMatrixDisplay,
)
from config import CONFUSION_MATRIX_PATH


def evaluate_model(pipeline, X_test, y_test):
    """Evaluates the model and saves the confusion matrix."""
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n--- Model Evaluation Report ---")
    print(
        classification_report(y_test, y_pred, target_names=["Not Flagged", "Flagged"])
    )

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Precision-Recall AUC Score: {pr_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Not Flagged", "Flagged"]
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix on Test Set")
    plt.savefig(CONFUSION_MATRIX_PATH)
    # plt.show()
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")
