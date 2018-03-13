import unittest
import numpy as np
import pandas as pd
import time
import logging
from homework_2.perceptron import train_model


class MyTestCase(unittest.TestCase):
    def test_perceptron(self):
        train_model()
        # learning_rate = 1
        # pos_data = pd.DataFrame({'x': [3, 4], 'y': [3, 3], 'label': [1, 1]})
        # neg_data = pd.DataFrame({'x': [1], 'y': [1], 'label': [-1]})
        # train_model(pos_data=pos_data, neg_data=neg_data, learning_rate=learning_rate, BATCH=True)
    @unittest.skip
    def test_run_time_for_different_learning_rate(self):
        run_times = []
        rates = np.linspace(0.1, 1, 10)
        for rate in rates:
            start = time.time()
            train_model(learning_rate=rate, BATCH=False)
            end = time.time()
            run_times.append(end - start)
            logging.warning("run time array:", run_times)
        import matplotlib.pyplot as plt
        plt.plot(rates, run_times)
        plt.show()


if __name__ == '__main__':
    unittest.main()
