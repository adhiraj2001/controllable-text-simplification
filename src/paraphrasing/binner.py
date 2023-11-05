import numpy as np


class GaussianBinner :
    def __init__(self, bin_count, gamma) -> None:
        self.bin_count = bin_count
        self.gamma = gamma
        self.means = None
        self.sigmas = None

    @staticmethod
    def gaussian_distance(val, mean, sigma) :
        return np.exp(-np.power(val - mean, 2.) / (2 * sigma * sigma))

    def create_bins(self, x) :
        feature_count = x.shape[1]
        self.means = []
        self.sigmas = []
        for feature in range(feature_count) :
            feature_vector = x[:, feature]
            feature_min, feature_max = np.min(feature_vector), np.max(feature_vector)
            bin_width = (feature_max - feature_min) / self.bin_count
            bins = np.arange(self.bin_count + 1) * bin_width + feature_min
            mean = np.array([bins[i] + bin_width / 2 for i in range(self.bin_count)])
            sigma = bin_width * self.gamma
            self.means.append(mean)
            self.sigmas.append(sigma)
        self.means = np.array(self.means)
        self.sigmas = np.array(self.sigmas)
    
    def generate_vectors(self, x) :
        x = np.array(x)
        x_reshaped = np.tile(x, (self.bin_count, 1, 1))
        means_reshaped = np.tile(self.means.T.reshape(self.bin_count, 1, -1), (1, x.shape[0], 1))
        sigmas_reshped = np.tile(self.sigmas, (self.bin_count, x.shape[0], 1))
        gaussian = self.gaussian_distance(x_reshaped, means_reshaped, sigmas_reshped)
        return np.roll(gaussian, 1, 0).reshape(x.shape[0], -1) 
            

class GaussianBinner1D(GaussianBinner) :
    def __init__(self, bin_count, gamma, min_val, max_val) :
        self.bin_count = bin_count
        self.gamma = gamma
        self.min_val = min_val
        self.max_val = max_val
        self.means = None
        self.sigma = None

        bin_width = (self.max_val - self.min_val) / self.bin_count
        bins = np.arange(self.bin_count + 1) * bin_width + self.min_val
        self.means = np.array([bins[i] + bin_width / 2 for i in range(self.bin_count)])
        self.sigma = bin_width * self.gamma

    def generate_vectors(self, x) :
        return GaussianBinner1D.gaussian_distance(x, self.means, self.sigma)