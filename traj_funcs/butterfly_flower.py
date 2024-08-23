import numpy as np

def generate_coordinates(timestep): # [0, 1]
    x = -2 + timestep * 1.5
    y = 0.5 - timestep * 0.5
    return np.array([x, y, 0]) # x y z