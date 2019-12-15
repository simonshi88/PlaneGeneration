import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from PlaneGeneration.Cell2D import Cell2D, Cell2DViewer
import random
import copy
import pandas

def RandomSeed(x):
    """Initialize the random and np.random generators.
    x: int seed
    """
    random.seed(x)
    np.random.seed(x)

def probability(p):
    """Returns True with probability `p`."""
    return np.random.random() < p


class CA_block(Cell2D):

    options = dict(mode='same', boundary='wrap')
    rule_g1 = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.int8)

    rule_g2 = np.array([[100, 1],
                       [1, 1]], dtype=np.int8)
    calculate = 0

    def __init__(self, n, m=None, p=0.5, p0=0.8, p1=0.8, random_seed=None):
        """Initializes the attributes.
                n: number of rows
                p: threshold on the initial entities percentage
                p0:probability in ground floor
                p1:probability in first or second floor
                random_seed: fixed selection of initial random data"""
        self.p_ground = p0
        self.p_first_second = p1
        if m is None:
            m = n
        RandomSeed(random_seed)
        """0 is constant units(roads), 1 is virtual spaces, 
        10 is the entities(buildings)"""
        choices = np.array([1, 10], dtype=np.int8)
        self.array = np.random.choice(choices, (n, m), p=[1-p, p])

        # Initializes the roads
        self.array[n * 2 // 3: n * 2 // 3 + 3, m * 2 // 3 - 3: m * 2 // 3] = 0
        self.array[n * 2 // 3: n * 2 // 3 + 1, : m * 2 // 3] = 0
        self.array[n // 3: n // 3 + 1, m // 3:] = 0
        self.array[: n * 2 // 3, n // 3: m // 3 + 1] = 0
        self.array[:n // 3, n * 3 // 4:m * 3 // 4 + 1] = 0
        self.array[n * 2 // 3:, m * 2 // 3:m * 2 // 3 + 1] = 0

        self.array_ground = np.zeros_like(self.array)
        self.array_first = np.zeros_like(self.array)
        self.array_second = np.zeros_like(self.array)

    def step(self):
        self.calculate += 1
        if self.calculate <= 5:
            self.rule_ground_floor()
        elif 5 < self.calculate <= 10:
            if np.all(self.array_first == 0):
                self.array_first = copy.deepcopy(self.array_ground)
            self.rule_first_floor()
        else:
            if np.all(self.array_second == 0):
                self.array_second = copy.deepcopy(self.array_first)
            self.rule_second_floor()

    def rule_ground_floor(self):
        self.array_ground = self.rule_general(self.array)

    def rule_first_floor(self):
        self.array_first = self.rule_general(self.array_first)
        self.array_first = self.rule_first_second(self.array_first)

    def rule_second_floor(self):
        self.array_second = self.rule_general(self.array_second)
        self.array_second = self.rule_first_second(self.array_second)
        a = self.array_second
        total = self.array + self.array_first + self.array_second
        rows, cols = a.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if (total[i, j] == 3 or total[i, j] == 0) and total[i-1, j] == 30:
                    if a[i+1, j] == 10:
                        a[i + 1, j] = 1
        self.array_second = a

    def rule_general(self, target_array):
        a = target_array
        c = correlate2d(a, self.rule_g1, mode='same', boundary='wrap')
        d = correlate2d(a, self.rule_g2, mode='same', boundary='wrap')
        a[c == 50] = 1

        if probability(self.p_ground):
            a[(d >= 100) & (d < 105)] = 10
        return a

    def rule_first_second(self, target_array):
        a = target_array
        rows, cols = a.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                state = a[i, j]
                # vertical direction
                neighbours1 = a[i, j - 1:j + 2]
                # horizontal direction"
                neighbours2 = a[i-1:i+2, j]
                # four corners
                neighbours3 = [a[i - 1, j + 1], a[i + 1, j + 1], a[i - 1, j - 1], a[i + 1, j - 1]]
                k1 = np.sum(neighbours1) - state
                k2 = np.sum(neighbours2) - state
                k3 = np.sum(neighbours3)
                if (state == 1 or 0) and probability(self.p_first_second):
                    if ((k1 == 20 and k2 != 20) or (k1 != 20 and k2 == 20)) and k3 < 40:
                        a[i, j] = 10
        return a

    def export_to_csv(self):
        export_data0 = pandas.DataFrame(self.array)
        export_data0.to_csv(r'.\initial.csv')

        export_data1 = pandas.DataFrame(self.array_ground)
        export_data1.to_csv(r'.\array_ground.csv', index=False, header=False)

        export_data2 = pandas.DataFrame(self.array_first)
        export_data2.to_csv(r'.\array_first.csv', index=False, header=False)

        export_data2 = pandas.DataFrame(self.array_second)
        export_data2.to_csv(r'.\array_second.csv', index=False, header=False)

class CA_block_Viewer(Cell2DViewer):
    colors = ['#c8c8c8', '#bae3f9','#007197']
    cmap = ListedColormap(colors)
    options = dict(interpolation='nearest', alpha=1,
                   vmin=0, vmax=2)

def main():
    n = 16
    m = 16
    cas = CA_block(n, m, 0.4, 0.8, 0.6, 18)
    viewer = CA_block_Viewer(cas)
    plt.figure(figsize=(12, 4))

    plt.subplot(141)

    viewer.draw(cas.array, grid=True)

    plt.subplot(142)
    viewer.step(5)
    viewer.draw(cas.array_ground,grid=True)

    plt.subplot(143)
    viewer.step(10)
    viewer.draw(cas.array_first, grid=True)

    plt.subplot(144)
    viewer.step(25)
    viewer.draw(cas.array_second, grid=True)

    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    plt.show()

    cas.export_to_csv()

if __name__ == '__main__':
    main()
