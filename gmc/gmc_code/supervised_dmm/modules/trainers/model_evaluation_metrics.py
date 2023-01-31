import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
import os
import wandb

def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    return (tp * (n / p) + tn) / (2 * n)


def eval_mosei(results, truths, logger, exclude_zero=False, vis_filename='default', mod='all', seed=0):
    if mod == [0]: mod_n = 'T'
    if mod == [1]: mod_n = 'A'
    if mod == [2]: mod_n = 'V'
    if mod == [0,1]: mod_n = 'TA'
    if mod == [0,2]: mod_n = 'TV'
    if mod == [1,2]: mod_n = 'AV'
    if mod == [0,1,2]: mod_n = 'TAV'

    out = open('./results/'+'0802_supervised_gmc'+'.txt', 'a')
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)
    acc = accuracy_score(binary_truth, binary_preds)
    
    wandb.log({f'mae_{mod_n}':mae,f'corr_{mod_n}':corr,f'fscore_{mod_n}':f_score,f'acc_{mod_n}':acc})

    print(vis_filename, seed, mod, mae, corr, f_score, acc, sep=',', file=out)
    # print("Seed: ", seed, file=out)
    # print("Test Modality: ", mod, file=out)
    # print("MAE: ", mae, file=out)
    # print("Correlation Coefficient: ", corr, file=out)
    # print("F1 score: ", f_score, file=out)
    # print("Accuracy: ", accuracy_score(binary_truth, binary_preds), file=out)
    # print("-" * 50, file=out)

    # Log results
    logger.log_metric("mae", mae)
    logger.log_metric("correlation", corr)
    logger.log_metric("f1_score", f_score)
    logger.log_metric("accuracy", acc)
    
    return mae, corr, f_score, acc


def eval_mosi(results, truths, logger, exclude_zero=False, vis_filename='default', mod='all', seed=0):
    return eval_mosei(results, truths, logger, exclude_zero)


def eval_iemocap(results, truths, logger, single=-1):
    emos = ["Neutral", "Happy", "Sad", "Angry"]
    if single < 0:
        test_preds = results.view(-1, 4, 2).cpu().detach().numpy()
        test_truth = truths.view(-1, 4).cpu().detach().numpy()

        for emo_ind in range(4):
            print(f"{emos[emo_ind]}: ")
            test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
            test_truth_i = test_truth[:, emo_ind]
            f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
            acc = accuracy_score(test_truth_i, test_preds_i)
            print("  - F1 Score: ", f1)
            print("  - Accuracy: ", acc)
    else:
        test_preds = results.view(-1, 2).cpu().detach().numpy()
        test_truth = truths.view(-1).cpu().detach().numpy()

        print(f"{emos[single]}: ")
        test_preds_i = np.argmax(test_preds, axis=1)
        test_truth_i = test_truth
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        print("  - F1 Score: ", f1)
        print("  - Accuracy: ", acc)



