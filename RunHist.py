
from collections import deque
import numpy as np

class RewardHist:
    def __init__(self, maxlen=100):
        self.mem = deque(maxlen=maxlen)
        self.last = 0

    def add(self, reward):
        self.mem.append(reward)

    def _nparr(self):
        return np.array(self.mem)

    def max(self):
        return self._nparr().max()

    def mean(self):
        return self._nparr().mean()

    def report(self):
        mean = self.mean()
        symbol = '▲' if mean > self.last else '▼' if mean < self.last else '-'
        print(f"Reward AVG: {mean:8.2f} | {symbol} {(mean - self.last):8.2f}")
        print(f"Best: {self.max()}")
        self.last = mean
