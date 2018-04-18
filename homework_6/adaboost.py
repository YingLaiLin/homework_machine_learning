# -*- coding: utf-8 -*-
import numpy as np
import math
from homework_6 import config


def main():
    train_data, labels = init_data()
    if config.is_supervised:
        features = ['x']
    else:
        features = ['身体', '业务', '潜力']
    assert len(train_data) > 0
    iterations = 10
    coefficients, weights = init_params(iterations, len(train_data))
    classifiers = init_classifiers(iterations)
    index = 1
    while index < iterations:
        print('---------------------------')
        classifier, error = search_best_classifier(train_data, labels,
                                                   weights[index])
        classifiers[index] = classifier
        print('best error: {}'.format(error))
        coefficient = math.log((1 - error) / error) / 2

        print('G{}(x) 的系数为{}'.format(index , coefficient))
        if classifier[2]:
            print('G{}(x): {} > {}  y = {},  {} < {}, y = {}'.format(index + 1,
                                                                     features[
                                                                         classifier[
                                                                             0].astype(
                                                                             int)],
                                                                     classifier[
                                                                         1],
                                                                     config.NEGATIVE_LABEL,
                                                                     features[
                                                                         classifier[
                                                                             0].astype(
                                                                             int)],
                                                                     classifier[
                                                                         1],
                                                                     config.POSITIVE_LABEL))
        else:
            print('G{}(x): {} < {}  y = {},  {} > {}, y = {}'.format(index + 1,
                                                                     features[
                                                                         classifier[
                                                                             0].astype(
                                                                             int)],
                                                                     classifier[
                                                                         1],
                                                                     config.NEGATIVE_LABEL,
                                                                     features[
                                                                         classifier[
                                                                             0].astype(
                                                                             int)],
                                                                     classifier[
                                                                         1],
                                                                     config.POSITIVE_LABEL))
        coefficients[index] = coefficient
        dim = classifier[0].astype(int)
        updated_weights = get_update_weights(train_data[:, dim], weights[index],
                                             classifier, coefficient, labels)
        weights[index + 1] = updated_weights
        # get boost adapter
        predictions_all = np.zeros(len(train_data))
        for classifier_index in range(1, index+1):
            classifier = classifiers[classifier_index]
            dim = classifier[0].astype(int)
            split_value = classifier[1]
            if classifier[2]:
                predictions_all += coefficients[classifier_index] * np.array(
                    list(map(lambda
                                 x: config.NEGATIVE_LABEL if x > split_value
                    else config.POSITIVE_LABEL,
                             train_data[:, dim])))
            else:
                predictions_all += coefficients[classifier_index] * np.array(
                    list(map(lambda
                                 x: config.NEGATIVE_LABEL if x < split_value
                    else config.POSITIVE_LABEL,
                             train_data[:, dim])))
        # predictions_all = np.array(list(map(
        #     lambda x: config.POSITIVE_LABEL if abs(
        #         x - config.POSITIVE_LABEL) < abs(
        #         x - config.NEGATIVE_LABEL) else config.NEGATIVE_LABEL,
        #     predictions_all)))
        print(predictions_all)
        predictions_all = np.array(list(map(
            lambda x: config.POSITIVE_LABEL if x > 0 else config.NEGATIVE_LABEL,
            predictions_all)))
        wrong_classified = np.sum(predictions_all != labels)
        print('分类误差点数量 {}'.format(np.sum(predictions_all != labels)))
        index += 1
        if not wrong_classified:
            break

    # output classifiers
    if config.can_output_classifiers:
        print('---------------------------')
        print('得到的树桩依次为:')
        for classifier_index in range(index):
            classifier = classifiers[classifier_index]
            if classifier[2]:
                print('data[{}] > {}, y = {}   data[{}] < {}, y = {}'.format(
                    classifier[0], classifier[1], config.NEGATIVE_LABEL,
                    classifier[0], classifier[1], config.POSITIVE_LABEL))
            else:
                print('data[{}] < {}, y = {}   data[{}] > {}, y = {}'.format(
                    classifier[0], classifier[1], config.NEGATIVE_LABEL,
                    classifier[0], classifier[1], config.POSITIVE_LABEL))


def init_data():
    if config.is_supervised:
        train = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
        labels = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    else:
        train = [[0, 1, 3], [0, 3, 1], [1, 2, 2], [1, 1, 3], [1, 2, 3],
                 [0, 1, 2], [1, 1, 2], [1, 1, 1], [1, 3, 1], [0, 2, 1]]
        labels = [-1, -1, -1, -1, -1, -1, 1, 1, -1, -1]
    train_data = np.array(train)
    labels = np.array(labels)
    return train_data, labels


def init_params(iterations, data_size):
    coefficients = np.zeros((iterations, 1))
    weights = np.ones((iterations + 1, data_size)) / data_size
    return coefficients, weights


'''
    使用第三个维度来控制 < 或 > 的使用.
'''


def init_classifiers(data_size):
    return np.zeros((data_size, 3))


def search_best_classifier(train_data, labels, weights, ):
    dim = len(train_data[0])
    best_classifier = np.array([0, 0, 0])
    best_error = 1
    for dim_index in range(dim):
        data = train_data[:, dim_index]
        best_error = 1
        split_values = get_available_values(list(set(data)))
        if config.can_output_available_values:
            print('dim:{},split_values: {}'.format(dim, split_values))
        for split_value in split_values:
            classifier = np.array([dim_index, split_value, 0])
            error = get_error(data, weights, classifier, labels)
            classifier_gt = np.array([dim_index, split_value, 1])
            error_gt = get_error(data, weights, classifier_gt, labels)
            if error_gt < error:
                error = error_gt
                classifier = classifier_gt
            if error < best_error:
                # print('切分点值: {}'.format(split_value))
                best_error = error
                best_classifier = classifier
    return best_classifier, best_error


def get_error(data, weights, classifier, labels):
    error = 0.0
    split_value = classifier[1]
    for index in range(len(data)):
        # 使用 >
        if classifier[2]:
            prediction = config.NEGATIVE_LABEL if data[
                                                      index] > split_value \
                else config.POSITIVE_LABEL
        else:
            prediction = config.NEGATIVE_LABEL if data[
                                                      index] < split_value \
                else config.POSITIVE_LABEL
        if prediction != labels[index]:
            error += weights[index]
    return error


def get_update_weights(data, weights, classifier, coefficient, labels):
    split_value = classifier[1]
    if classifier[2]:
        predictions = np.array(list(map(lambda
                                            x: config.NEGATIVE_LABEL if x >
                                                                        split_value else config.POSITIVE_LABEL,
                                        data)))
    else:
        predictions = np.array(list(map(lambda
                                            x: config.NEGATIVE_LABEL if x <
                                                                        split_value else config.POSITIVE_LABEL,
                                        data)))
    x = np.exp(-coefficient * labels * predictions)
    z = np.dot(weights, x)
    updated_weights = weights * x / z
    return updated_weights


def get_available_values(split_values):
    return [sum(split_values[k:k + 2]) / len(split_values[k:k + 2]) for k in
            range(0, len(split_values) - 1, 1)]


if __name__ == "__main__":
    main()
