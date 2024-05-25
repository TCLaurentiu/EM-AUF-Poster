import numpy as np
import skimage
from PIL import Image
import matplotlib.pyplot as plt

def read_image(path):
  return skimage.io.imread(path)

def rgb2lab(rgb):
  
  [y, x, junk] = rgb.shape

  def labF(inp):
    out = inp ** (1/3) * (inp > 0.008856)
    out += (7.787 * inp + 16/116)*(inp <= .008856)
    return out

  flattened = rgb.transpose(1, 0, 2).reshape(-1, 3) / 255

  XYZn = np.array([0.9505, 1.0000, 1.0888]).reshape(-1, 1)

  XYZ = np.array([
    [0.412453, 0.357580, 0.180423], 
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227]
    ]).dot(flattened.T)

  Lstar = 116*labF(XYZ[1, :]) - 16

  astar = 500 * (labF(XYZ[0, :] / XYZn[0]) - labF(XYZ[1, :]))

  bstar = 200 * (labF(XYZ[1, :]) - labF(XYZ[2, :] / XYZn[2]))

  return Lstar.reshape(x, y).T, astar.reshape(x, y).T, bstar.reshape(x, y).T