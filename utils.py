import numpy as np

def manhattan_distance(point1, point2):
    return np.abs(point2[0] - point1[0]) + np.abs(point2[1] - point1[1])

def minMaxScale(min,max,input):
    return (input - min) / (max - min)

def get_turtle_dimensions(t):
    stretch_wid, stretch_len, outline_width = t.shapesize()
    base_size = 20  # Default base size of the turtle shape (20 pixels)
    
    width = stretch_len * base_size
    height = stretch_wid * base_size

    return height,width