import turtle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from game import Game
from objects.snake import Snake
from objects.board import Board
from objects.food import Food
from utils import manhattan_distance,minMaxScale

inputSize = 14
numClasses = 4
class Solver (nn.Module):
    def __init__(self):
        super(Solver,self).__init__()

        self.network = nn.Sequential(
            nn.Linear(inputSize, 16),
            nn.Sigmoid(),
            nn.Linear(16, 32),
            nn.Sigmoid(),
            nn.Linear(32, numClasses)
        )


    def getNearestObjectDistanceFromRight(self,snake:Snake, rightWallPos):
        segmentsInRow = snake.getAllSegmentsInRow(snake.head.ycor())
        segmentsInRow = [x.xcor() for x in segmentsInRow if x.xcor() > snake.head.xcor()]
        segmentsInRow.append(rightWallPos)
        return np.min([ np.abs(x - snake.head.xcor()) for x in segmentsInRow])
    

    def getNearestObjectDistanceFromLeft(self,snake:Snake, leftWallPos):
        segmentsInRow = snake.getAllSegmentsInRow(snake.head.ycor())
        segmentsInRow = [x.xcor() for x in segmentsInRow if x.xcor() <  snake.head.xcor()]
        segmentsInRow.append(leftWallPos)

        return np.min([ np.abs(x - snake.head.xcor()) for x in segmentsInRow])
    
    def getNearestObjectDistanceFrodown(self,snake:Snake, upWallPos):
        segmentsInCol = snake.getAllSegmentsInCol(snake.head.xcor())
        segmentsInCol = [x.ycor() for x in segmentsInCol if x.ycor() <  snake.head.ycor()]
        segmentsInCol.append(upWallPos)
        return np.min([ np.abs(x - snake.head.ycor()) for x in segmentsInCol])
    
    def getNearestObjectDistanceFroUp(self,snake:Snake, upWallPos):
        segmentsInCol = snake.getAllSegmentsInCol(snake.head.xcor())
        segmentsInCol = [x.ycor() for x in segmentsInCol if x.ycor() >  snake.head.ycor()]
        segmentsInCol.append(upWallPos)
        return np.min([ np.abs(x - snake.head.ycor()) for x in segmentsInCol])

    
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

        return np.array([
            headXPos,
            headYpos,
            snakeLength,
            foodXPos,
            foodYPos,
            distanceToFood,
            leftWallDistance,
            rightWallDistance,
            upWallDistance,
            downWallDistance,
            nearestObjectDistanceFromLeft,
            nearestObjectDistanceFromRight,
            nearestObjectDistanceFromup,
            nearestObjectDistanceFromdown
        ])


    def scaleVector(self,vector,game: Game):
        verticalWallsMaxDistance = np.abs(game.board.leftWallPos-game.board.rightWallPos)
        horizontalWallsMaxDistance = np.abs(game.board.upWallPos-game.board.downWallPos)
        foodMaxDistance = manhattan_distance((game.board.leftWallPos,game.board.upWallPos),(game.board.rightWallPos,game.board.downWallPos))

        return np.array([
            minMaxScale(game.board.leftWallPos,game.board.rightWallPos,vector[0]),
            minMaxScale(game.board.upWallPos,game.board.downWallPos,vector[1]),
            minMaxScale(1,400,vector[2]),
            minMaxScale(game.board.leftWallPos,game.board.rightWallPos,vector[3]),
            minMaxScale(game.board.upWallPos,game.board.downWallPos,vector[4]),
            minMaxScale(0,foodMaxDistance,vector[5]),
            minMaxScale(0,verticalWallsMaxDistance,vector[6]),
            minMaxScale(0,verticalWallsMaxDistance,vector[7]),
            minMaxScale(0,horizontalWallsMaxDistance,vector[8]),
            minMaxScale(0,horizontalWallsMaxDistance,vector[9]),
            minMaxScale(0,verticalWallsMaxDistance,vector[10]),
            minMaxScale(0,verticalWallsMaxDistance,vector[11]),
            minMaxScale(0,horizontalWallsMaxDistance,vector[12]),
            minMaxScale(0,horizontalWallsMaxDistance,vector[13]),
        ])

    def forward(self, game):
        x = self.createVector(game)
        x= self.scaleVector(x,game)
        x = torch.tensor(x, dtype=torch.float32)
        x= self.network(x)
        return nn.functional.softmax(x)
    
    def action(self,game):
        x = self.forward(game)
        return torch.argmax(x).item()
    
        