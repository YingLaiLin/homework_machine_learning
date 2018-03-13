import numpy as np
import matplotlib.pyplot as plt

CURVES_NUMBER_TO_DRAW = 10


def main():
    # TODO do something
    generate_data()
    m = [50, 40, 40, 35, 35, 30, 20, 20, 10, 10]
    n = [30, 20, 20, 15, 15, 10, 10, 8, 5, 5]
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC-Curve")
    for index in range(CURVES_NUMBER_TO_DRAW):
        change_scores(m[index], n[index])
        FPR = []
        TPR = []
        positive, negative = 60, 40
        positive_scores.extend(negative_scores)
        thresholds = positive_scores.copy()
        thresholds.sort()
        for threshold in thresholds[:-1]:
            positive_predicted = list(
                map(classify, positive_scores, [threshold] * 60))
            TPR.append(sum(list(
                map(lambda x, y: 1 if x == y else 0, positive_predicted,
                    positive_labels))) / positive)
            negative_predicted = map(classify, negative_scores,
                                     [threshold] * 40)
            FPR.append(sum(list(
                map(lambda x, y: 1 if x == y else 0, negative_predicted,
                    positive_labels))) / negative)
        plt.plot(FPR, TPR)
    plt.show()


def classify(x, y):
    if x >= y:
        return 1
    return 0


# 前60个数据类别为 +1, 后40个数据类别为 -1, score ∈ [0,1]
def generate_data():
    global positive_labels
    positive_labels = [1] * 60
    global negative_labels
    negative_labels = [-1] * 40
    global positive_scores
    positive_scores = [0.7] * 60
    global negative_scores
    negative_scores = [0.3] * 40


def change_scores(m, n):
    random_scores = np.random.random(m + n)
    positive_scores[:m] = random_scores[:m]
    negative_scores[:n] = random_scores[m + 1:]


if __name__ == "__main__":
    main()
