import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix,
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataloader.

    Returns:
        (avg_loss, accuracy, preds, labels, probs) where probs are softmax probabilities.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(preds, labels, class_names):
    """Compute overall and per-class classification metrics."""
    metrics = {}

    # Overall metrics
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['precision_macro'] = precision_score(labels, preds, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(labels, preds, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)

    # Per-class metrics
    precision_per = precision_score(labels, preds, average=None, zero_division=0)
    recall_per = recall_score(labels, preds, average=None, zero_division=0)
    f1_per = f1_score(labels, preds, average=None, zero_division=0)

    for i, cname in enumerate(class_names):
        metrics[f'precision_{cname}'] = precision_per[i]
        metrics[f'recall_{cname}'] = recall_per[i]
        metrics[f'f1_{cname}'] = f1_per[i]

    return metrics


def compute_roc_auc(labels, probs, class_names):
    """Compute ROC curves and AUC scores for multiclass classification.

    Returns:
        (fpr, tpr, roc_auc) dicts with per-class, micro, and macro entries.
    """
    num_classes = len(class_names)
    labels_bin = label_binarize(labels, classes=list(range(num_classes)))

    fpr = {}
    tpr = {}
    roc_auc = {}

    # Per-class ROC/AUC
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average: aggregate all classes
    fpr["micro"], tpr["micro"], _ = roc_curve(labels_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average: mean of per-class AUCs
    roc_auc["macro"] = np.mean([roc_auc[i] for i in range(num_classes)])

    return fpr, tpr, roc_auc


def plot_roc_curves(fpr, tpr, roc_auc, class_names, title="ROC Curves", save_path=None):
    """Plot ROC curves for all classes plus micro-average."""
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i, cname in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                 label=f'{cname} (AUC = {roc_auc[i]:.4f})')

    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"], color='black', lw=2, linestyle='--',
             label=f'Micro-avg (AUC = {roc_auc["micro"]:.4f})')

    plt.plot([0, 1], [0, 1], 'k:', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(preds, labels, class_names, title="Confusion Matrix", save_path=None):
    """Plot a confusion matrix."""
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells with counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


def print_metrics(metrics, split_name, class_names):
    """Pretty print metrics."""
    print(f"\n{'='*50}")
    print(f"{split_name} Metrics")
    print(f"{'='*50}")
    print(f"Overall Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro Precision:   {metrics['precision_macro']:.4f}")
    print(f"Macro Recall:      {metrics['recall_macro']:.4f}")
    print(f"Macro F1:          {metrics['f1_macro']:.4f}")
    print(f"\nPer-class metrics:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 51)
    for cname in class_names:
        print(f"{cname:<15} {metrics[f'precision_{cname}']:<12.4f} "
              f"{metrics[f'recall_{cname}']:<12.4f} {metrics[f'f1_{cname}']:<12.4f}")
