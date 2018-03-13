import numpy as np


def main():
    # TODO do something
    X = np.array(
        [0.000000, 0.1111, 0.2222, 0.3333, 0.4444, 0.5556, 0.6667, 0.7778,
         0.8889, 1])
    Y = np.array([
        [0.0008, 0.6419, 0.9849, 0.8655, 0.3423, -0.3426, -0.8655, -0.9841,
         -0.6411, -0.0002]]).T

    order_list = [0, 1, 3, 9]
    for order in order_list:
        vander_X = get_expansion_with_order(order, X)
        w = get_solution(vander_X, Y)
        print("When M is %d, the res is %s" % (order, w))


def get_expansion_with_order(order, basic_X):
    vander_X = np.vander(basic_X, order, increasing=True)
    return vander_X


def get_solution(X, Y):
    xTx = X.T.dot(X)
    if 0 == np.linalg.det(xTx):
        print('xTx is a singular matrix')
        return
    return np.linalg.inv(xTx).dot(X.T).dot(Y)


if __name__ == "__main__":
    main()
