import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve


def model_performance(History, title: str = 'Model Performance'):
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Plotting the training and validation accuracies
    axs[0].plot(History.history['accuracy'], color='red', lw=4, label='Training')
    axs[0].plot(History.history['val_accuracy'], color='blue', lw=4, label='Validation')

    # Labeling the x and y axes
    axs[0].set_xlabel('Epochs', fontsize=16)
    axs[0].set_ylabel('Accuracy', fontsize=16)

    # Setting the fontsize and color for x and y tick labels
    axs[0].tick_params(axis='both', labelsize=14)

    # Setting the title
    axs[0].set_title(y=1.025, label='Accuracy of the Model', fontsize=18, fontweight='bold')

    # Plotting the legend
    axs[0].legend(fontsize=12, framealpha=0.5)

    # Plotting the training and validation losses
    axs[1].plot(History.history['loss'], color='red', lw=4, label='Training')
    axs[1].plot(History.history['val_loss'], color='blue', lw=4, label='Validation')

    # Labeling the x and y axes
    axs[1].set_xlabel('Epochs', fontsize=16)
    axs[1].set_ylabel('Loss', fontsize=16)

    # Setting the fontsize and color for x and y tick labels
    axs[1].tick_params(axis='both', labelsize=14)

    # Setting the title
    axs[1].set_title(y=1.025, label='Loss of the Model', fontsize=18, fontweight='bold')

    # Plotting the legend
    axs[1].legend(fontsize=12, framealpha=0.5)

    fig.suptitle(y=1.025, t=title, fontsize=22, fontweight='bold')

    plt.show()


def predictions(model, data_generator, preprocessor: bool = False):
    pred = model.predict(data_generator, verbose=0)

    if preprocessor:
        pred_labels = []
        for i in range((data_generator.n // data_generator.batch_size) + 1):
            _, y = data_generator.next()
            pred_labels.extend(y)
    else:
        pred_labels = [int(y.numpy()[0]) for _, y in data_generator.unbatch()]

    return pred, pred_labels


def plot_auc_roc(model, balanced_data_generator, imbalanced_data_generator, preprocessor: bool = False):
    if preprocessor:
        balanced_pred, balanced_pred_labels = predictions(model, balanced_data_generator, preprocessor)
        imbalanced_pred, imbalanced_pred_labels = predictions(model, imbalanced_data_generator, preprocessor)
    else:
        balanced_pred, balanced_pred_labels = predictions(model, balanced_data_generator)
        imbalanced_pred, imbalanced_pred_labels = predictions(model, imbalanced_data_generator)

    balanced_roc_auc_score = roc_auc_score(balanced_pred_labels, balanced_pred)
    imbalanced_roc_auc_score = roc_auc_score(imbalanced_pred_labels, imbalanced_pred)

    fpr_balanced, tpr_balanced, threshold_balanced = roc_curve(balanced_pred_labels, balanced_pred)
    fpr_imbalanced, tpr_imbalanced, threshold_imbalanced = roc_curve(imbalanced_pred_labels, imbalanced_pred)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    axs[0].set_axisbelow(True)
    axs[0].grid(which='major', linestyle='--', alpha=0.75)

    axs[1].set_axisbelow(True)
    axs[1].grid(which='major', linestyle='--', alpha=0.75)

    ns_fpr = [0, 1]
    ns_tpr = [0, 1]

    axs[0].plot(fpr_balanced, tpr_balanced, lw=4, color='red', alpha=0.75,
                label='AUC-ROC on Balanced Data: {:.4f}'.format(balanced_roc_auc_score))
    axs[0].plot(ns_fpr, ns_tpr, linestyle='--', color='orange', lw=3,
                label='ROC curve of an Average Classifier (AUC: 0.5)')

    axs[0].tick_params(axis='both', labelsize=14)
    axs[0].set_xlabel('False Positive Rate', fontsize=16)
    axs[0].set_ylabel('True Positive Rate', fontsize=16)
    axs[0].legend(fontsize=12)
    axs[0].set_title('On Balanced Test Data', fontsize=16, fontweight='bold')

    axs[1].plot(fpr_imbalanced, tpr_imbalanced, lw=4, color='blue', alpha=0.75,
                label='AUC-ROC on Imbalanced Data: {:.4f}'.format(imbalanced_roc_auc_score))
    axs[1].plot(ns_fpr, ns_tpr, linestyle='--', color='orange', lw=3,
                label='ROC curve of an Average Classifier (AUC: 0.5)')

    axs[1].tick_params(axis='both', labelsize=14)
    axs[1].set_xlabel('False Positive Rate', fontsize=16)
    axs[1].set_ylabel('True Positive Rate', fontsize=16)
    axs[1].legend(fontsize=12)
    axs[1].set_title('On Imbalanced Test Data', fontsize=16, fontweight='bold')

    fig.suptitle(y=1.05, t='ROC Curve of the {}'.format(model.name), fontsize=18, fontweight='bold')

    plt.show()


def plot_confusion_matrix(model, balanced_data_generator, imbalanced_data_generator, preprocessor: bool = False):
    if preprocessor:
        balanced_pred, balanced_pred_labels = predictions(model, balanced_data_generator, preprocessor)
        imbalanced_pred, imbalanced_pred_labels = predictions(model, imbalanced_data_generator, preprocessor)
    else:
        balanced_pred, balanced_pred_labels = predictions(model, balanced_data_generator)
        imbalanced_pred, imbalanced_pred_labels = predictions(model, imbalanced_data_generator)

    balanced_pred_class, imbalanced_pred_class = [np.array([balanced_pred >= 0.5]).astype(int).flatten(),
                                                  np.array([imbalanced_pred >= 0.5]).astype(int).flatten()]

    # Plotting Confusion Matrix
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    bal_conf_matrix = confusion_matrix(balanced_pred_labels, balanced_pred_class)
    names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    counts = ['{}'.format(value) for value in bal_conf_matrix.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in bal_conf_matrix.flatten() / np.sum(bal_conf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts, percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.set(font_scale=1.2)
    sns.heatmap(bal_conf_matrix / np.sum(bal_conf_matrix), fmt='', cmap='Reds', ax=axs[0], annot=labels)
    axs[0].set_xticklabels(['Damaged', 'Undamaged'])
    axs[0].set_yticklabels(['Damaged', 'Undamaged'])
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[0].set_xlabel('Predicted Labels', fontsize=14)
    axs[0].set_ylabel('True Labels', fontsize=14)
    axs[0].set_title('On Balanced Test Data', fontsize=16, fontweight='bold')

    imb_conf_matrix = confusion_matrix(imbalanced_pred_labels, imbalanced_pred_class)
    names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    counts = ['{}'.format(value) for value in imb_conf_matrix.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in imb_conf_matrix.flatten() / np.sum(imb_conf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts, percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.set(font_scale=1.2)
    sns.heatmap(imb_conf_matrix / np.sum(imb_conf_matrix), fmt='', cmap='Reds', ax=axs[1], annot=labels)
    axs[1].set_xticklabels(['Damaged', 'Undamaged'])
    axs[1].set_yticklabels(['Damaged', 'Undamaged'])
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    axs[1].set_xlabel('Predicted Labels', fontsize=14)
    axs[1].set_ylabel('True Labels', fontsize=14)
    axs[1].set_title('On Imbalanced Test Data', fontsize=16, fontweight='bold')

    fig.suptitle(y=1, t='Confusion Matrices of the {}'.format(model.name), fontsize=18, fontweight='bold')

    plt.show()


def evaluation_metrics(model, balanced_data_generator, imbalanced_data_generator, preprocessor: bool = False):
    balancedLoss, balancedAccuracy = model.evaluate(balanced_data_generator, verbose=0)
    imbalancedLoss, imbalancedAccuracy = model.evaluate(imbalanced_data_generator, verbose=0)

    balancedAccuracy, imbalancedAccuracy = ['{:.2f}%'.format(round(balancedAccuracy, 4) * 100),
                                            '{:.2f}%'.format(round(imbalancedAccuracy, 4) * 100)]

    balancedLoss, imbalancedLoss = [round(balancedLoss, 3), round(imbalancedLoss, 3)]

    if preprocessor:
        balanced_pred, balanced_pred_labels = predictions(model, balanced_data_generator, preprocessor)
        imbalanced_pred, imbalanced_pred_labels = predictions(model, imbalanced_data_generator, preprocessor)
    else:
        balanced_pred, balanced_pred_labels = predictions(model, balanced_data_generator)
        imbalanced_pred, imbalanced_pred_labels = predictions(model, imbalanced_data_generator)

    balanced_pred_class, imbalanced_pred_class = [np.array([balanced_pred >= 0.5]).astype(int).flatten(),
                                                  np.array([imbalanced_pred >= 0.5]).astype(int).flatten()]

    bal_precision_score = round(precision_score(balanced_pred_labels, balanced_pred_class), 3)
    imb_precision_score = round(precision_score(imbalanced_pred_labels, imbalanced_pred_class), 3)

    bal_recall_score = round(recall_score(balanced_pred_labels, balanced_pred_class), 3)
    imb_recall_score = round(recall_score(imbalanced_pred_labels, imbalanced_pred_class), 3)

    bal_conf_matrix = confusion_matrix(balanced_pred_labels, balanced_pred_class)
    imb_conf_matrix = confusion_matrix(imbalanced_pred_labels, imbalanced_pred_class)

    bal_fpr, bal_fnr = [bal_conf_matrix[0][1] / (bal_conf_matrix[0][0] + bal_conf_matrix[0][1]),
                        bal_conf_matrix[1][0] / (bal_conf_matrix[1][1] + bal_conf_matrix[1][0])]

    bal_fpr, bal_fnr = '{:.2f}%'.format(round(bal_fpr, 4) * 100), '{:.2f}%'.format(round(bal_fnr, 4) * 100)

    imb_fpr, imb_fnr = [imb_conf_matrix[0][1] / (imb_conf_matrix[0][0] + imb_conf_matrix[0][1]),
                        imb_conf_matrix[1][0] / (imb_conf_matrix[1][1] + imb_conf_matrix[1][0])]

    imb_fpr, imb_fnr = '{:.2f}%'.format(round(imb_fpr, 4) * 100), '{:.2f}%'.format(round(imb_fnr, 4) * 100)

    metrics_dict = {'Metrics': ['Accuracy', 'Loss', 'Precision Score', 'Recall Score',
                                'False-Positive Rate', 'False-Negative Rate'],
                    'Balanced Dataset': [balancedAccuracy, balancedLoss, bal_precision_score,
                                         bal_recall_score, bal_fpr, bal_fnr],
                    'Imbalanced Dataset': [imbalancedAccuracy, imbalancedLoss, imb_precision_score,
                                           imb_recall_score, imb_fpr, imb_fnr]}

    return pd.DataFrame.from_dict(metrics_dict)
