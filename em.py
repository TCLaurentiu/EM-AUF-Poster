import numpy as np
from numpy import linalg

import matplotlib.pyplot as plt
from matplotlib import cm

np.set_printoptions(suppress = True, precision = 3)

from scipy.stats import multivariate_normal

from tqdm import trange

class Density():

  def prepare_ax(self, ax, *args, **kwargs):
    if ax is None:
      fig = plt.figure()
      ax = fig.add_subplot(1, 1, 1, *args, **kwargs)
      ax.cla()
    
    return ax

  def plot_1d(self, xmin = -3, xmax = 3, point_count = 1000, ax = None, scale = 1, *args, **kwargs):
    """
    Plots `density_function` on the interval [xmin, xmax] by taking `point_count` values
    
    ax is used to plot multiple graphs on the same canvas
    the density_function is optionally multipled by the parameter `scale`
    
    extra arguments are passed to the plot function
    """
    x = np.expand_dims(np.linspace(xmin, xmax, point_count), 1)

    ax = self.prepare_ax(ax)
    
    ax.plot(x, self(x) * scale, *args, **kwargs)
    ax.set_xlabel("x")
    ax.set_ylabel("p(x)")

    return ax

  def plot_2d(self, xmin = -3, xmax = 3, ymin = -3, ymax = 3, point_count = 100, ax = None, scale = 1, *args, **kwargs):
    
    xr, yr, zr = self.get_data_for_2d_plot(xmin, xmax, ymin, ymax, point_count)

    ax = self.prepare_ax(ax, projection = "3d")
    
    ax.plot_surface(xr, yr, zr, cmap=cm.coolwarm)

    return ax

  def get_data_for_2d_plot(self, xmin = -3, xmax = 3, ymin = -3, ymax = 3, point_count = 1000):
    x = np.linspace(xmin, xmax, point_count)
    y = np.linspace(ymin, ymax, point_count)
    x, y = np.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()

    pos = np.empty((x.shape[0],) + (2, ))
    pos[:, 0] = x.flatten()
    pos[:, 1] = y.flatten()

    z = self(pos).reshape((point_count, point_count))
    xr = x.reshape((point_count, point_count))
    yr = y.reshape((point_count, point_count))
    zr = z.reshape((point_count, point_count))

    return xr, yr, zr


  def plot_contour(self, xmin = -3, xmax = 3, ymin = -3, ymax = 3, point_count = 100, ax = None, scale = 1, *args, **kwargs):
    xr, yr, zr = self.get_data_for_2d_plot(xmin, xmax, ymin, ymax, point_count)

    ax = self.prepare_ax(ax)
    
    ax.contour(xr, yr, zr, cmap=cm.coolwarm, levels = 10)

    if hasattr(self, "mean"):
      ax.plot(self.mean[0], self.mean[1], 'go', alpha = 0.8)

    return ax

  def plot(self, *args, **kwargs):
    if self.dim == 1:
      return self.plot_1d(*args, **kwargs)
    
    if self.dim == 2:
      return self.plot_2d(*args, **kwargs)
      
    raise NotImplemented

  def likelihood(self, data):
    return np.prod(self(data))

  def log_likelihood(self, data):
    return np.sum(np.log(self(data)))

  def sample(self, size):
    return np.random.multivariate_normal(mean = self.mean, cov = self.variance, size = size)

class GaussianDensity(Density):
  def __init__(self, mean, variance):
    if type(mean) in [int, float]:
      mean = np.array([mean])
    if type(variance) in [int, float]:
      variance = np.array([[variance]])

    assert mean.shape[0] == variance.shape[0] == variance.shape[1], "mean must be an n-dimensional vector and variance must be an n x n square matrix"

    self.dim = mean.shape[0]

    self.mean = mean
    self.variance = variance

  def __call__(self, x):
    if type(x) in [int, float]:
      x = np.array([[x]])
      
    values = multivariate_normal.pdf(x, self.mean, self.variance, allow_singular = True)

    if values.shape == ():
      values = np.array([values])

    return values

  @classmethod
  def standard(cls, dim = 1):
    return cls(np.zeros(dim), np.eye(dim))

  def __str__(self):
    return f'mean {self.mean} and variance {self.variance.flatten()}'

class Mixture(Density):
  def __init__(self, densities, weights):
    if type(weights) == list:
      weights = np.array(weights)
    assert len(densities) == len(weights), "must provide one weight for each density"
    assert np.sum(weights) == 1, "weights must sum up to 1"
    assert not np.any(weights < 0), "weights must be >= 0"
    
    assert np.all(np.array([comp.dim for comp in densities]) == densities[0].dim), "all mixture components must have the same dimensionality"

    self.dim = densities[0].dim

    self.densities = densities
    self.weights = weights

  def plot_mixture(self, xmin = -10, xmax = 10, ymin = -10, ymax = -10, plot_components = True, plot_mixture = True):
    densities, weights = self.densities, self.weights

    if self.dim == 1:
      plot_fn = self.plot_1d
      plot_fn_str = "plot_1d"
    else:
      plot_fn = self.plot_contour
      plot_fn_str = "plot_contour"

    ax = self.prepare_ax(None)

    if plot_mixture:
      if plot_fn_str == "plot_1d":
        ax = plot_fn(xmin = xmin, xmax = xmax, label = "GMM", c = "black", zorder = 10, ax = ax)
      else:
        ax = plot_fn(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, label = "GMM", c = "black", zorder = 10, ax = ax)

    if plot_components:
      for id, (density, weight) in enumerate(zip(densities, weights)):
        if plot_fn_str == "plot_1d":
          density.plot_1d(xmin = xmin, xmax = xmax, scale = weight, linestyle = "dashed", ax = ax, label = f'Component {id}')
        else:
          density.plot_contour(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, scale = weight, linestyle = "dashed", ax = ax, label = f'Component {id}')

    ax.legend()

    return ax

  def __call__(self, x):
    return np.sum([weight * density(x) for density, weight in zip(self.densities, self.weights)], axis = 0)

  def __len__(self):
    return len(self.densities)

  @classmethod
  def standard(cls, dim, count):
    return cls([GaussianDensity.standard(dim) for _ in range(count)], np.ones(count) / count)

  @classmethod
  def random(cls, dim, count):
    return cls([GaussianDensity(mean = np.random.rand(dim), variance = np.random.rand() * np.eye(dim)) for _ in range(count)], np.ones(count) / count)

  def __str__(self):
    return '\n'.join([str(density) + f' with weight {weight}' for density, weight in zip(self.densities, self.weights)])

class EM:
  def __init__(self, mixture, data):
    self.mixture = mixture
    self.data = data
    self.cache = dict()
  
  def compute_responsabilities(self):
    component_count = len(self.mixture)
    data_size = len(self.data)
    weights = self.mixture.weights

    top = np.array([self.mixture.densities[i](self.data) for i in range(component_count)]).T
    bottom = np.sum(top * weights, axis = 1)
    
    responsabilities = top * weights / bottom[:, None]
    self.cache["responsabilities"] = responsabilities
    return responsabilities

  def compute_total_responsabilities(self):
    return np.sum(self.cache["responsabilities"], axis = 0)

  def compute_new_means(self):
    responsabilities = self.cache["responsabilities"]
    return responsabilities.T.dot(self.data) / np.sum(responsabilities, axis = 0)[:, None]

  def perform_mean_update(self):
    new_means = self.compute_new_means()
    for density, new_mean in zip(self.mixture.densities, new_means):
      density.mean = new_mean

  def compute_new_variances(self):
    responsabilities = self.cache["responsabilities"]
    total_responsabilities = self.compute_total_responsabilities()
    # extract the means from the components
    means = np.array([component.mean for component in self.mixture.densities])
    variances = np.array([np.sum(responsabilities[:, i].reshape(-1, 1, 1) * np.einsum("... b, d ...", (self.data - means[i]), (self.data - means[i]).T), axis = 0) / total_responsabilities[i] for i in range(len(self.mixture))])
    return variances

  def perform_variance_update(self):
    new_variances = self.compute_new_variances()
    for density, new_variance in zip(self.mixture.densities, new_variances):
      density.variance = new_variance

  def compute_new_weights(self):
    total_responsabilities = self.compute_total_responsabilities()
    return total_responsabilities / len(self.data)

  def perform_weight_update(self):
    new_weights = self.compute_new_weights()
    self.mixture.weights = new_weights

  def fit(self, steps = 100, eps = 0.01, force_steps = True, return_likelihoods = False):
    initial_nll = -self.mixture.log_likelihood(self.data)
    likelihoods = [initial_nll]
    for i in trange(steps):
      self.compute_responsabilities()
      self.perform_mean_update()
      self.perform_variance_update()
      self.perform_weight_update()
      updated_nll = -self.mixture.log_likelihood(self.data)
      likelihoods.append(updated_nll)

      if np.allclose(initial_nll, updated_nll, atol=eps, rtol=0):
        if not force_steps:
          break

      initial_nll = updated_nll

    if return_likelihoods:
      return likelihoods

  def plot_fit(self, plot_mixture = True, plot_components = True, xmin = -10, xmax = 10, ymin = -10, ymax = 10):
    ax = self.mixture.plot_mixture(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, plot_mixture = plot_mixture, plot_components = plot_components)
    if self.mixture.dim == 1:
      ax.scatter(self.data, y = np.zeros_like(self.data), c = "blue", zorder = 100)
    else:
      ax.scatter(self.data[:, 0], self.data[:, 1], alpha = 0.3, c = "black")

  def likelihood(self):
    return self.mixture.likelihood(self.data)

  def log_likelihood(self):
    return self.mixture.log_likelihood(self.data)

  def plot_colored_by_responsabilities(self):
    plt.scatter(self.data[:, 0], self.data[:, 1], alpha = 1, c = self.cache["responsabilities"])

def plot_nll(nll):
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.cla()
  ax.ticklabel_format(style='plain') 
  ax.set_xlabel("Iteration")
  ax.set_ylabel("Negative log-likelihood")
  ax.plot(nll, marker="o")

