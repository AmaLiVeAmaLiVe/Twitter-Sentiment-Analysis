import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# function for evaluating the model
def evaluate_model(model, subset, y_target):
    # predict values
    y_pred = model.predict(subset)
    y_pred_proba = model.predict_proba(subset)

    # print the evaluation metrics for the dataset
    print(classification_report(y_target, y_pred))

    # calculating accuracy score, F1-score, ROC AUC score
    accuracy = metrics.accuracy_score(y_target, y_pred)
    f1_score = metrics.f1_score(y_target, y_pred)
    roc_auc = metrics.roc_auc_score(y_target, y_pred_proba[:, 1])

    print(f"Accuracy: {accuracy}")
    print(f"F1 score: {f1_score:.6f}")
    print(f'ROC AUC score for: {roc_auc}')

    # compute and plot the confusion matrix
    cf_matrix = confusion_matrix(y_target, y_pred)
    categories = ['Negative', 'Positive']
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = [f"{round(value, 2) * 100}%" for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1} - {v2}' for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='', xticklabels=categories, yticklabels=categories)
    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)
    plt.show()

    # plotting ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y_target, y_pred_proba[:, 1], pos_label=1)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # return the metrics of the model
    return accuracy, f1_score, roc_auc