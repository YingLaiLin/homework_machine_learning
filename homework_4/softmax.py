# -*- coding: utf-8 -*-

from functools import wraps
from homework_4 import config
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_curve
from tqdm import tqdm


def cal_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        args = func(*args, **kwargs)
        end = time.time()
        print("{} cost {} s ".format(func.__name__, end - start))
        return args

    return wrapper


@cal_time
def load_train_data_from_file(dir_name):
    cnt = 0
    train_data = None
    for root, dirs, files in os.walk(dir_name):
        train_data_size = len(files)
        train_data = np.zeros((train_data_size, config.data_size))
        for file in files:
            train_data[cnt, :] = file2matrix(dir_name + file)
            cnt += 1
    return train_data  # (1934,1024)


def file2matrix(filename):
    with open(filename) as f:
        data = f.readlines()
        len_of_array = len(data)
        mat = np.zeros((len_of_array, config.image_size))  # 32 * 32
        index = 0
        for line in data:
            mat[index, :] = [int(x) for x in line.split("\n")[0]]
            index += 1

    return mat.reshape(-1)  # (1024)


@cal_time
def load_label_from_file(dir_name):
    labels = None
    for root, dirs, files in os.walk(dir_name):
        train_data_size = len(files)
        labels = np.zeros((train_data_size, config.label_classes))
        cnt = 0
        for filename in files:
            label_index = filename.rfind("_")
            if -1 != label_index:
                labels[cnt, int(filename[label_index - 1])] = 1  # 0 - 9
            cnt += 1
    return labels  # (1934,1024), (1934, 10)


@cal_time
def train_model(train_set, labels):
    w = init_model()
    for epoch in tqdm(range(config.iteration_round)):
        batch_data = train_set[:]
        predictions = get_predictions(w, train_set)
        sub = predictions - labels[:].T
        for cnt in range(config.label_classes):
            #            t = np.zeros((config.label_classes, IMAGE_SIZE ** 2))
            #            for index in range(len(batch_data)):
            #                for label_index in range(config.label_classes):
            #                    t[label_index, :] += sub[label_index,
            # index] * batch_data[
            #                        index]

            weight = get_deriviative(sub[cnt, :], batch_data)

            w[cnt, :] -= get_weight(weight, w[cnt, :],
                                    config.learning_rate,
                                    optimizer=config.gradient_descent)
        score = show_performance(get_predictions(w, train_set), labels)
        print('epoch {}: score is {}'.format(epoch, score))
    return w


def show_performance(predictions, labels):
    return accuracy_score(np.argmax(predictions, axis=0),
                          np.argmax(labels.T, axis=0))


# TODO 修改权重更新值的方法
def get_weight(w, pre_w, learning_rate, optimizer=config.gradient_descent):
    if "gd" == optimizer:
        return learning_rate * w
    elif "re" == optimizer:
        return learning_rate * (w + config.regularization_lambda * pre_w)
    elif "rd" == optimizer:
        return learning_rate * w + np.random.normal(w.shape[0])
    else:
        raise ("not such optimizer name named {}".format(optimizer))


@cal_time
def init_model():
    return np.random.rand(config.label_classes, config.data_size)  # (10, 1024)


# TODO get real deriviative
"""
    optimizer : available parameter is 'gradient', 'regularization', 'random'
"""


def get_deriviative(sub, batch_data):
    t = np.zeros(config.data_size)
    for index in range(len(batch_data)):
        t += sub[index] * batch_data[index]
    return t


def get_prediction_prob(w, train_set=None):
    if train_set is None:
        raise ValueError("train data is empty, please check it!")
    prediction_values = np.dot(w, train_set.T)
    prediction_values = (prediction_values - prediction_values.min()) / (
            prediction_values.max() - prediction_values.min())
    for col_index in range(prediction_values.shape[1]):
        prediction_values[:, col_index] = prediction_values[:, col_index] / sum(
            np.exp(prediction_values[:, col_index]))
    return prediction_values  # nlabels * ndata (10 * 1934)


def get_predictions(w, train_set=None):
    predictions_prob = get_prediction_prob(w, train_set)
    predictions = np.zeros(predictions_prob.shape)
    for column_index in range(predictions_prob.shape[1]):
        predictions[:, column_index] = predictions_prob[:,
                                       column_index] // np.max(
            predictions_prob[:, column_index])
    return predictions


def get_train_data():
    train_dir_name = "/Users/leon/Documents/machine_learning" \
                     "/homework_4/softmax_datasets/trainingDigits/"

    train_set = load_train_data_from_file(train_dir_name)
    labels = load_label_from_file(train_dir_name)
    if labels is None or train_set is None:
        raise ("File {} Not Found!".format(train_dir_name))
    return train_set, labels


# TODO 增加多类的 ROC 曲线绘制
def show_roc_curve(labels, predictions_prob):
    fpr = dict()
    tpr = dict()
    for row_index in range(config.label_classes):
        fpr[row_index], tpr[row_index], _ = roc_curve(labels.T[row_index, :],
                                                      predictions_prob[
                                                      row_index, :], )
    plt.figure()
    for i in range(config.label_classes):
        plt.plot(fpr[i], tpr[i], label="digit {}".format(i))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


def main():
    train_set, labels = get_train_data()
    w = train_model(train_set, labels)
    predictions_prob = get_prediction_prob(w, train_set)
    show_roc_curve(labels, predictions_prob)
    test_dir_name = "/Users/leon/Documents/machine_learning" \
                    "/homework_4/softmax_datasets/testDigits/"
    test_set = load_train_data_from_file(test_dir_name)
    print(np.argmax(get_predictions(w, test_set), axis=0))

if __name__ == "__main__":
    main()
