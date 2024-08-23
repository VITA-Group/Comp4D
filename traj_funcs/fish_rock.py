import numpy as np

def generate_coordinates(t):
    z = 1.0 * np.sin(t*np.pi/2)
    x = -1.5 * np.cos(t*np.pi/2)
    return np.array([x, 0, z])