import turtle
import numpy as np
import random

class Food():
    def __init__(self,minX,maxX,minY,maxY):

        self.minX = minX
        self.minY = minY
        self.maxX = maxX
        self.maxY = maxY

        self.food = turtle.Turtle()
        self.food.speed(0)
        self.food.shape("circle")
        self.food.color("red")
        self.food.penup()
        self.randomLocaation()

    def __call__(self):
        return self.food

    def randomLocaation(self):
        x = random.randint(self.minX, self.maxX)
        y = random.randint(self.minY,self.maxY )
        self.food.goto(x,y)