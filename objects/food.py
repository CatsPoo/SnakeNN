import turtle
import numpy as np
import random

class Food():
    def __init__(self,minX,maxX,minY,MaxY):
        self.minX = minX
        self.minY = minY
        self.maxX = maxX
        self.maxY = MaxY

        self.food = turtle.Turtle()
        self.food.speed(0)
        self.food.shape("circle")
        self.food.color("red")
        self.food.penup()
        self.food.goto(0,100)

    def __call__(self):
        return self.food

    def randomLocaation(self):
        x = random.randint(self.minX, self.maxX)
        y = random.randint(self.minY,self.maxY )
        self.food.goto(x,y)