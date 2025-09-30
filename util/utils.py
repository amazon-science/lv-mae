import numpy as np
import torch
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from itertools import cycle



def save_data(all_embeddings, all_ids, args, path_output_embeds, iter):
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_ids = torch.cat(all_ids, dim=0)
    langbind_data = {"embeddings": all_embeddings, "video_ids": all_ids ,"args": args}
    torch.save(langbind_data, os.path.join(path_output_embeds, f'data_langbind_{iter}.pt'))
    print(f"Saved batch {iter}. Batch embeddings size: {all_embeddings.shape}")


def compute_ret_metrics(x):
    # Emanuel:
    # This is my correction to the original 'compute_metrics' function from LangBind repo.
    # In the original computation, they subtract the diagonal *scores* from the sorted *scores*.
    # This leads to having possible more than one 'zero' per row. We see that len(ind) > 1000 (we got 1012) which
    #   does not make sense.
    # This funciton fixes this issue: it subtracted the diagonal *indexes* from the sorted *indexes*.
    sort_idx = np.argsort(-x, axis=1)
    d = np.arange(len(x))
    d = d[:, np.newaxis]
    ind = sort_idx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    # metrics["cols"] = [int(i) for i in list(ind)]
    return metrics


def compute_class_metrics(sim_matrix, labels, classes):
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}
    label_indices = torch.tensor([class_to_index[label] for label in labels])
    targets = torch.nn.functional.one_hot(label_indices, len(classes))

    # Top-1 Accuracy
    top1_pred = sim_matrix.argmax(dim=1)    # sim_matrix is the prediction matrix
    top1_correct = (top1_pred == label_indices).sum().item()
    top1_accuracy = top1_correct / len(labels)

    # Top-5 Accuracy
    top5_pred = sim_matrix.topk(5, dim=1).indices
    top5_correct = top5_pred.eq(label_indices.view(-1, 1).expand_as(top5_pred)).sum().item()
    top5_accuracy = top5_correct / len(labels)

    return top1_accuracy, top5_accuracy


def compute_multilabel_metrics(sim_matrix, labels, classes, path_output=None):
    '''
    Here we assume a multi-label classification task, where only one label presents in a given sample, while
    other lables are assumed to not being present in the sample. So, for each sample we have one positive and N-1
    negative labels.
    '''
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}
    label_indices = torch.tensor([class_to_index[label] for label in labels])
    targets = torch.nn.functional.one_hot(label_indices, len(classes)).numpy()
    sim_matrix = sim_matrix.numpy()

    ap, map = mAP(targets, sim_matrix)

    # Thresholding
    preds = thresholding(sim_matrix, threshold=0.12)
    precision, recall, f1 = precision_recall(targets, preds)
    num_tp, num_fp, num_pos_gt, num_samples = get_numbers(targets, preds)

    metrics = {'map': map, 'ap': ap, 'precision': precision, 'recall': recall, 'num_tp': num_tp, 'num_fp': num_fp,
               'num_pos_gt': num_pos_gt, 'num_samples': num_samples}

    # Plot precision and recall
    plot_precision_recall_curves(targets, sim_matrix, classes, path_output)

    return metrics


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return ap, 100 * ap.mean()


def precision_recall(targs, preds, comp_sklearn=False):
    epsilon = 1e-8

    # make sure that preds has hard values
    preds_ = (preds > 0.).astype(int)

    precision_values = np.zeros((preds.shape[1]))
    recall_values = np.zeros((preds.shape[1]))
    f1_values = np.zeros((preds.shape[1]))
    for k in range(targs.shape[1]):
        if comp_sklearn:
            # sklearn metric
            precision = precision_score(targs[:, k], preds_[:, k])
            recall = recall_score(targs[:, k], preds_[:, k])
            f1 = f1_score(targs[:, k], preds_[:, k])
        else:
            # precision, recall and F1
            tp = sum(targs[:, k] * preds_[:, k])
            fp = sum( (1-targs[:, k]) * preds_[:, k])
            fn = sum(targs[:, k] * (1-preds_[:, k]))
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            f1 = 2 * (precision * recall) / (precision + recall + epsilon)

        precision_values[k] = precision
        recall_values[k] = recall
        f1_values[k] = f1

    return precision_values, recall_values, f1_values


def get_numbers(targs, preds):

    # make sure that preds has hard values
    preds_ = (preds > 0.).astype(int)

    num_tp = np.zeros((preds.shape[1]))
    num_fp = np.zeros((preds.shape[1]))

    num_pos_gt = np.zeros((preds.shape[1]))
    num_samples = targs.shape[0]

    for k in range(targs.shape[1]):
        # precision, recall and F1
        tp = sum(targs[:, k] * preds_[:, k])
        fp = sum((1-targs[:, k]) * preds_[:, k])
        total_pos = sum(targs[:, k])

        num_tp[k] = tp
        num_fp[k] = fp
        num_pos_gt[k] = total_pos

    return num_tp, num_fp, num_pos_gt, num_samples


def thresholding(sim_matrix, threshold):

    predictions = sim_matrix > threshold

    return predictions


def plot_precision_recall_curves(targets, preds, classes, path_output):

    path_save = os.path.join(path_output, 'precision_recall')
    os.makedirs(path_save, exist_ok=True)

    n_classes = targets.shape[1]
    # colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    colors = cycle(['darkorange', 'black', 'red', 'blue', 'green'])
    save_n_classes = len(classes)
    for i, color in zip(range(n_classes), colors):
        if i > save_n_classes:
            break
        plt.figure(figsize=(10, 8))
        precision, recall, _ = precision_recall_curve(targets[:, i], preds[:, i])
        plt.plot(recall, precision, color=color, lw=3, label=f'{classes[i]}')

        plt.xlabel('Recall', fontsize=16)
        plt.ylabel('Precision', fontsize=16)
        plt.title('Precision-Recall', fontsize=16)
        plt.legend(loc='best', fontsize=16)
        plt.grid()
        # plt.show()

        path_save_file = os.path.join(path_save, f'{classes[i]}.png')
        plt.savefig(path_save_file)