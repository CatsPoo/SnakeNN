# Simple Snake Game in Python 3 for Beginners
# By @TokyoEdTech

import turtle
import time
import random
import numpy as np
from objects.board import Board
from objects.snake import Snake
from objects.food import Food
from gameMode import GamkeMode
from utils import manhattan_distance,get_turtle_dimensions


score = 0
high_score = 0



class Game:
    screenPadding = 30
    high_score = 0

    def __init__(self,gameMode,wn,boardWidth,boardHeight,boardOffset,col = 0,row = 0,solver=None):
        self.board = Board(boardHeight,boardWidth,boardOffset,row,col)
        self.snake = None
        self.food = None
        self.solver = solver
        self.delay = 0.1
        self.gameMode = gameMode
        self._score =0

        self.wn = wn

        self.resetGame()
        # comtrols and keyboard settings bindings
        if(self.gameMode == GamkeMode.HUMAN):
            self.setupControls()

        if(self.gameMode == GamkeMode.AI):
            self.randomDirectionStart()
            self.snake.move()
    
    @property
    def score(self):
        return self._score
    
    @score.setter
    def score(self,val):
        self._score = val

    def resetGame(self):
        self.snake = Snake(self.board.centerX,self.board.centerY)
        self.food = Food(self.board.leftWallPos,self.board.rightWallPos,self.board.downWallPos,self.board.upWallPos)
        self.delay = 0.1
        self.score = 0

    def setupControls(self):
        self.wn.listen()  # Use wn instead of self.wn
        self.wn.onkeypress(self.snake.go_up, "w")
        self.wn.onkeypress(self.snake.go_down, "s")
        self.wn.onkeypress(self.snake.go_left, "a")
        self.wn.onkeypress(self.snake.go_right, "d")

    def randomDirectionStart(self):
        dir = random.randint(0,4)
        self.convertIndexToAction(dir)
        

    def convertIndexToAction(self,index):
        if(index == 0): self.snake.go_up()
        elif(index == 1): self.snake.go_down()
        elif(index == 2): self.snake.go_left()
        elif(index == 3): self.snake.go_right()


    def runGame(self):
        # Main game loop
        while True:
            self.wn.update()

            if(self.gameMode == GamkeMode.AI):
                self.convertIndexToAction(self.solver.action(self))


            # Check for a collision with the border
            if (self.snake.head.xcor() - self.snake.getWidth() < self.board.leftWallPos or
                self.snake.head.xcor() + self.snake.getWidth()//2> self.board.rightWallPos or
                self.snake.head.ycor() - 2*self.snake.getHeight()< self.board.downWallPos or
                self.snake.head.ycor() + self.snake.getHeight()//2 > self.board.upWallPos):
                time.sleep(1)
                self.snake.head.goto(self.board.centerX,self.board.centerY)
                self.snake.head.direction = "stop"
                
                # Clear the segments list
                self.board.writeScore(self.score,Game.high_score)
                if(self.gameMode == GamkeMode.AI):
                    return
                self.endGame()
                # Reset the delay
                self.resetGame()


            # Check for a collision with the food
            if self.snake.head.distance(self.food()) < 20:
                # Move the food to a random spot
                self.food.randomLocaation()

                # Add a segment
                self.snake.addSegment()

                # Shorten the delay
                self.delay -= 0.001

                # Increase the score
                self.score += 10

                if self.score > Game.high_score:
                    Game.high_score = self.score
                
                self.board.writeScore(self.score,Game.high_score)

            # Move the end segments first in reverse order
            self.snake.moveSegments()
            self.snake.move()    

            # Check for head collision with the body segments
            if(self.snake.checkForSelfCollition()):
                time.sleep(1)
                self.snake.head.goto(0,0)
                self.snake.head.direction = "stop"
                
                if(self.gameMode == GamkeMode.AI):
                    return
                self.endGame()
                self.resetGame()


                
                # Update the score display
                self.board.writeScore(self.score,Game.high_score)

            time.sleep(self.delay)
    
    def endGame(self):
        self.food().hideturtle()
        self.food().clear()
        self.snake.head.hideturtle()
        self.snake.head.clear()
        [seg.clear() for seg in self.snake.segments]
        self.snake.kill()