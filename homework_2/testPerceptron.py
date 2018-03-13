import unittest
import numpy as np
import pandas as pd
from homework_2.perceptron import train_perceptron, plot_data


class MyTestCase(unittest.TestCase):
    def test_perceptron(self):
        w = np.array([0] * 2)
        b = 0
        learning_rate = 1
        pos_data = pd.DataFrame({'x': [3, 4], 'y': [3, 3], 'label': [1, 1]})
        neg_data = pd.DataFrame({'x': [1], 'y': [1], 'label': [-1]})
        train_data = np.array(pd.concat([pos_data, neg_data]))
        w, b = train_perceptron(train_data, w, b, learning_rate)
        plot_data(pos_data, neg_data, w, b)


if __name__ == '__main__':
    unittest.main()
