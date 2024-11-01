import turtle
import numpy as np

class Board:
    def __init__(self,width,height):
        self.leftWallPos = -(width//2)
        self.rightWallPos = width//2
        self.upWallPos = height//2
        self.downWallPos = -(height//2)
        
    def getWidth(self):
        return np.abs(self.leftWallPos-self.rightWallPos)

    def getHeight(self):
        return np.abs(self.upWallPos-self.downWallPos)

