import numpy as np
import matplotlib.pyplot as plt


class DensityEstimation:

    def __init__(self):
        self._ndim = None
        self._bin_width = None
        self._domain = None
        self._p = None

    def run(self, x, h):
        self._ndim = x.ndim
        self._bin_width = h
        if self._ndim == 1:
            self._solver_1d(x)
        elif self._ndim == 2:
            self._solver_2d(x)
        else:
            raise NotImplementedError()
        return self._p, self._domain

    def plot(self):
        if self._domain is not None and self._p is not None and self._bin_width is not None:
            if self._ndim == 1:
                plt.bar(self._domain[:, 0], self._p, width=self._bin_width)
            elif self._ndim == 2:
                plt.imshow(self._p, vmin=0, vmax=0.15)
        else:
            print("Please run DensityEstimation.run(x, h) first.")

    def _solver_1d(self, data):
        num_of_data = len(data)
        min_val = np.min(data)
        max_val = np.max(data)
        edges = np.arange(min_val, max_val, self._bin_width)
        if edges[-1] < max_val:
            edges = np.append(edges, max_val)
        num_of_bin = len(edges) - 1
        self._domain = np.zeros((num_of_bin, 2))
        for idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            self._domain[idx] = [left, right]
        self._p = np.zeros(num_of_bin)
        for idx in range(num_of_bin):
            if idx == num_of_bin - 1:
                condition = np.logical_and((data >= self._domain[idx, 0]),
                                           (data <= self._domain[idx, 1]))
            else:
                condition = np.logical_and((data >= self._domain[idx, 0]),
                                           (data < self._domain[idx, 1]))
            num_of_observation = len(data[condition])
            self._p[idx] = self._compute_p(num_of_observation, num_of_data)

    def _solver_2d(self, data):
        num_of_data = len(data)
        x_dim = data[:, 0]
        y_dim = data[:, 1]
        x_min_val = np.min(x_dim)
        x_max_val = np.max(x_dim)
        y_min_val = np.min(y_dim)
        y_max_val = np.max(y_dim)
        x_edges = np.arange(x_min_val, x_max_val, self._bin_width)
        y_edges = np.arange(y_min_val, y_max_val, self._bin_width)
        if x_edges[-1] < x_max_val:
            x_edges = np.append(x_edges, x_max_val)
        if y_edges[-1] < y_max_val:
            y_edges = np.append(y_edges, y_max_val)
        y_edges = y_edges[::-1]
        x_num_of_bins = len(x_edges) - 1
        y_num_of_bins = len(y_edges) - 1
        self._domain = np.zeros((x_num_of_bins, y_num_of_bins, 4))
        for x_idx, (x_left, x_right) in enumerate(zip(x_edges[:-1],
                                                      x_edges[1:])):
            for y_idx, (y_left,
                        y_right) in enumerate(zip(y_edges[1:], y_edges[:-1])):
                self._domain[x_idx, y_idx] = [x_left, y_left, x_right, y_right]
        self._p = np.zeros((y_num_of_bins, x_num_of_bins))
        for x_idx in range(x_num_of_bins):
            for y_idx in range(y_num_of_bins):
                if x_idx == x_num_of_bins - 1 and y_idx == y_num_of_bins - 1:
                    condition_x = np.logical_and(
                        (data[:, 0] >= self._domain[x_idx, y_idx, 0]),
                        (data[:, 0] <= self._domain[x_idx, y_idx, 2]))
                    condition_y = np.logical_and(
                        (data[:, 1] >= self._domain[x_idx, y_idx, 1]),
                        (data[:, 1] <= self._domain[x_idx, y_idx, 3]))
                else:
                    condition_x = np.logical_and(
                        (data[:, 0] >= self._domain[x_idx, y_idx, 0]),
                        (data[:, 0] < self._domain[x_idx, y_idx, 2]))
                    condition_y = np.logical_and(
                        (data[:, 1] >= self._domain[x_idx, y_idx, 1]),
                        (data[:, 1] < self._domain[x_idx, y_idx, 3]))

                num_of_observation = len(data[np.logical_and(
                    condition_x, condition_y)])
                self._p[y_idx, x_idx] = self._compute_p(num_of_observation,
                                                        num_of_data)

    def _compute_p(self, num_of_observation, num_of_data):
        return num_of_observation / num_of_data / self._bin_width
