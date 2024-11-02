import turtle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from game import Game
from objects.snake import Snake
from objects.board import Board
from objects.food import Food
from utils import manhattan_distance

inputSize = 4
numClasses = 4
class Solver (nn.Module):
    def __init__(self):
        super(Solver,self)

        self.network = nn.Sequential(
            nn.Linear(inputSize, 16),
            nn.Sigmoid(),
            nn.Linear(16, 32),
            nn.Sigmoid(),
            nn.Linear(32, numClasses)
        )


    def getNearestObjectDistanceFromRight(self,snake:Snake, rightWallPos):
        segmentsInRow = snake.getAllSegmentsInRow()
        segmentsInRow = [x for x in segmentsInRow if x.xcor() > snake.head.xcor()]
        segmentsInRow.append(rightWallPos)
        return np.min([ np.abs(x.xcor() - snake.head.xcor()) for x in segmentsInRow])
    

    def getNearestObjectDistanceFromLeft(self,snake:Snake, leftWallPos):
        segmentsInRow = snake.getAllSegmentsInRow()
        segmentsInRow = [x for x in segmentsInRow if x.xcor() <  snake.head.xcor()]
        segmentsInRow.append(leftWallPos)
        return np.min([ np.abs(x.xcor() - snake.head.xcor()) for x in segmentsInRow])
    
    def getNearestObjectDistanceFrodown(self,snake:Snake, upWallPos):
        segmentsInCol = snake.getAllSegmentsInCol()
        segmentsInCol = [x for x in segmentsInCol if x.ycor() <  snake.head.ycor()]
        segmentsInCol.append(upWallPos)
        return np.min([ np.abs(x.xcor() - snake.head.xcor()) for x in segmentsInCol])
    
    def getNearestObjectDistanceFroUp(self,snake:Snake, upWallPos):
        segmentsInCol = snake.getAllSegmentsInCol()
        segmentsInCol = [x for x in segmentsInCol if x.ycor() >  snake.head.ycor()]
        segmentsInCol.append(upWallPos)
        return np.min([ np.abs(x.xcor() - snake.head.xcor()) for x in segmentsInCol])

    
    def createVector(self,game:Game):
        leftWallDistance = np.abs(game.board.leftWallPos - game.snake.head.xcor())
        rightWallDistance = np.abs(game.board.rightWallPos - game.snake.head.xcor())
        upWallDistance = np.abs(game.board.upWallPos - game.snake.head.ycor())
        downWallDistance = np.abs(game.board.downWallPos - game.snake.head.ycor())

        nearestObjectDistanceFromLeft = self.getNearestObjectDistanceFromLeft(game.snake,game.board.leftWallPos)
        nearestObjectDistanceFromRight = self.getNearestObjectDistanceFromRight(game.snake,game.board.rightWallPos)
        nearestObjectDistanceFromup = self.getNearestObjectDistanceFroUp(game.snake,game.board.upWallPos)
        nearestObjectDistanceFromdown = self.getNearestObjectDistanceFrodown(game.snake,game.board.downWallPos)

        snakeLength = len(game.snake.segments)+1

        distanceToFood = manhattan_distance((game.snake.head.xcor(),game.snake.head.ycor()),(game.food().xcor(),game.food().ycor()))
        foodXPos = game.food().xcor()
        foodYPos = game.food().ycor()

        headXPos = game.snake.head.xcor()
        headYpos = game.snake.head.ycor()



    def forward(self, game):
        x = self.createVector(game)
        return self.network(x)
    
    def loss(self,foodDistance, leftWallDistance,rightWallDistance,upWallDistance,downWallDistance):
        return (leftWallDistance + rightWallDistance + upWallDistance + downWallDistance) - (4 * foodDistance) 
        