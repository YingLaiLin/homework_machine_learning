import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets



def main():
    # X = np.array([[1, 1], [2, 2], [2, 0], [0, 0], [1, 0], [0, 1]])
    # y = np.array([1, 1, 1, -1, -1, -1])
    X = np.array([[1, 1], [2, 2], [2, 0], [0, 0], [1, 0], [0, 1]])
    y = np.array([1, 1, 1, -1, -1, -1])
    C = 100  # SVM regularization parameter
    clf = svm.SVC(C=C, kernel='linear')
    clf = clf.fit(X, y)
    print('Support Vectors: ', clf.support_vectors_)
    print('Weighs: %s intercept: %s' % (clf.coef_, clf.intercept_))
    print('Dual Weights: %s' % clf.dual_coef_)
    # title for the plots
    visualize_svm(X, y, clf)


def visualize_svm(X, y, clf):
    X0, X1 = X[:, 0], X[:, 1]
    plt.scatter(X0, X1, c=y, cmap=plt.cm.Paired, s=30)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none')
    plt.show()


if __name__ == "__main__":
    main()
