import os
import math

PI = 3.1415926


def main():
    # TODO do something
    x_list = [0, 0.1111, 0.2222, 0.3333, 0.4444, 0.5556, 0.6667, 0.7778, 0.8889,
              1]
    res_list = []
    for x in x_list:
        res_list.append(tylor(x))
    for res in res_list:
        print(res)


"""
    将 y = sin(2pi * x) 展开到3阶, 对函数进行拟合
"""


def tylor(x):
    correlations = [2 * PI, 0, -4 / 3 * PI * PI * PI]
    parameter = x
    ans = 0.0
    print(correlations)
    for index in range(2):
        ans += correlations[index] * parameter
        parameter *= x
    return ans


if __name__ == "__main__":
    main()
