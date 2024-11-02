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
from utils import manhattan_distance


score = 0
high_score = 0



class Game:
    screenPadding = 30
    high_score = 0

    def __init__(self,gameMode,wn):
        self.board = Board(600,600)
        self.snake = Snake()
        self.food = Food(self.board.leftWallPos,self.board.rightWallPos,self.board.downWallPos,self.board.upWallPos)

        self.delay = 0.1

        self.gameMode = gameMode
        self.wn = wn
        self.wn.title("Snake Game by @TokyoEdTech")
        self.wn.bgcolor("lime")
        self.wn.setup(width=self.board.getWidth()+ Game.screenPadding, height=self.board.getHeight() +Game.screenPadding)
        self.wn.tracer(0) # Turns off the screen updates

        # setup board borders
        self.border_turtle = turtle.Turtle()
        self.setupBorders()

        #text visualization
        self.pen = turtle.Turtle()
        self.setupPen()

        # comtrols and keyboard settings bindings
        if(self.gameMode == GamkeMode.HUMAN):
            self.setupControls()

        if(self.gameMode == GamkeMode.AI or self.gameMode == GamkeMode.TRAINING):
            self.randomDirectionStart()
            self.snake.move()
    
    def setupBorders(self):
        self.border_turtle.hideturtle()  # Hide the turtle arrow icon
        self.border_turtle.speed(0)      # Fastest drawing speed
        self.border_turtle.color("black")  # Set the border color to black
        self.border_turtle.pensize(3)    # Set the border thickness

        # Move to starting position (top-left corner of the border)
        self.border_turtle.penup()
        self.border_turtle.goto(self.board.leftWallPos, self.board.upWallPos)  # Adjust to be just inside the screen size
        self.border_turtle.pendown()

        # Draw the border (a square or rectangle)
        for i in range(2):
            self.border_turtle.forward(self.board.getWidth())  # Adjust for your board size
            self.border_turtle.right(90)
            self.border_turtle.forward(self.board.getHeight())
            self.border_turtle.right(90)
    
    def setupPen(self):
        self.pen.speed(0)
        self.pen.shape("square")
        self.pen.color("white")
        self.pen.penup()
        self.pen.hideturtle()
        self.pen.goto(0, 260)
        self.writeScore(0,0)

    
    def writeScore(self,score,highScore):
        self.pen.clear()
        self.pen.write("Score: {}  High Score: {}".format(score,highScore), align="center", font=("Courier", 24, "normal"))

    def setupControls(self):
        self.wn.listen()  # Use wn instead of self.wn
        self.wn.onkeypress(self.snake.go_up, "w")
        self.wn.onkeypress(self.snake.go_down, "s")
        self.wn.onkeypress(self.snake.go_left, "a")
        self.wn.onkeypress(self.snake.go_right, "d")

    def randomDirectionStart(self):
        dir = random.randint(0,4)
        if(dir == 0): self.snake.go_up()
        elif(dir == 1): self.snake.go_down()
        elif(dir == 2): self.snake.go_left()
        elif(dir == 3): self.snake.go_right()


    def runGame(self):
        # Main game loop
        while True:
            self.wn.update()

            leftWallDistance = np.abs(self.board.leftWallPos - self.snake.head.xcor())
            rightWallDistance = np.abs(self.board.rightWallPos - self.snake.head.xcor())
            upWallDistance = np.abs(self.board.upWallPos - self.snake.head.ycor())
            downWallDistance = np.abs(self.board.downWallPos - self.snake.head.ycor())

            nearestWallDistance = np.min([leftWallDistance,rightWallDistance,upWallDistance,downWallDistance])
            nearestWallDistance = np.max([nearestWallDistance,0])

            # Check for a collision with the border
            if (nearestWallDistance == 0):
                time.sleep(1)
                self.snake.head.goto(0,0)
                self.snake.head.direction = "stop"
                
                # Clear the segments list
                self.snake.kill()
                # Reset the delay
                self.delay = 0.1
                self.writeScore(self.snake.score,Game.high_score)
                if(self.gameMode != GamkeMode.HUMAN):
                    break


            # Check for a collision with the food
            if self.snake.head.distance(self.food()) < 20:
                # Move the food to a random spot
                self.food.randomLocaation()

                # Add a segment
                self.snake.addSegment()

                # Shorten the delay
                self.delay -= 0.001

                # Increase the score
                self.snake.score += 10

                if self.snake.score > Game.high_score:
                    Game.high_score = self.snake.score
                
                self.writeScore(self.snake.score,Game.high_score)

            # Move the end segments first in reverse order
            self.snake.moveSegments()
            self.snake.move()    

            # Check for head collision with the body segments
            if(self.snake.checkForSelfCollition()):
                time.sleep(1)
                self.snake.head.goto(0,0)
                self.snake.head.direction = "stop"
                self.snake.kill()

                if(self.gameMode != GamkeMode.HUMAN):
                    break

                # Reset the score
                Snake.score = 0

                # Reset the delay
                self.delay = 0.1
                
                # Update the score display
                self.writeScore(self.snake.score,Game.high_score)

            time.sleep(self.delay)