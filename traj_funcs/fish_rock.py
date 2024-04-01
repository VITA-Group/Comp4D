import numpy as np

def generate_coordinates(t): # [0, 1]
    z = 0.8 * np.sin(t*np.pi/2)
    x = -1.5 * np.cos(t*np.pi/2)
    return np.array([x, 0, z]) # x y z