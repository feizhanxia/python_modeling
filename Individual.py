import numpy.random as npr
import numpy as np


class Individual:
    def __init__(self, n):
        self.data = npr.random(n)
        self.n = n

    def mutation(self):
        son = Individual(self.n)
        son.data = self.data.copy()
        pos = npr.randint(self.n)
        son.data[pos] += npr.random() - 0.5
        return son
