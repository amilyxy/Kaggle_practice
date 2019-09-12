# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# matplotlib.use("Qt5Agg")
# print('001230')
# plt.plot(np.arange(100))
# plt.show()

import itertools
version1 = "1.2.3"
version2 = "1.1"


def plusOne(digits)
    L = len(digits)
    out = []
    num = 0
    for i in range(L):
        num += pow(10, i)*digits[L-1-i]
    num += 1
    while True:
        out.append(num%10)
        num = num//10
        if num == 0:
            return out[::-1]


print(compareVersion( version1, version2))

