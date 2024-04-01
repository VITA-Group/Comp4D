import numpy as np

def generate_coordinates(timestep): # [0, 1]
    x = -2 + timestep*1.7
    y = 0.6 - timestep*0.4
    return np.array([x, y, 0]) # x y z