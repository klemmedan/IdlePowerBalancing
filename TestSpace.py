import matplotlib.pyplot as plt
import numpy as np
import math

s = 1e5
m = 125


def current_log(x):
    return np.floor(500 / m * np.log10([1 + i for i in x]))


def delayed_log(y):
    if y < s:
        return 0
    else:
        return np.floor(500/(m-np.log10(s)) * np.log10(1+y/s))


x = np.linspace(0, 126, 1000)
x = [10**i for i in x]
plt.semilogx(x, current_log(x))
plt.semilogx(x, [delayed_log(y) for y in x])
plt.show()


