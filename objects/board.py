import turtle
import numpy as np

class Board:
    def __init__(self,height,width,boardOffset,row,col):

        self.leftWallPos  = boardOffset+ (width * col)
        self.rightWallPos = boardOffset+ (width * (col+1)) - 1
        self.downWallPos = boardOffset+ (height * row)
        self.upWallPos =  boardOffset+ (height * (row+1)) -1

        self.centerX = boardOffset + self.leftWallPos + (width//2)
        self.centerY = boardOffset + self.downWallPos + (height //2)

        #text visualization
        self.pen = turtle.Turtle()
        self.setupPen()
        self.printWalls()
        
    def getWidth(self):
        return np.abs(self.leftWallPos-self.rightWallPos) + 1

    def getHeight(self):
        return np.abs(self.upWallPos-self.downWallPos) + 1
    
    def printWalls(self):
        walls_turtle = turtle.Turtle()
        walls_turtle.hideturtle()  # Hide the turtle arrow icon
        walls_turtle.speed(0)      # Fastest drawing speed
        walls_turtle.color("black")  # Set the border color to black
        walls_turtle.pensize(3)    # Set the border thickness

        # Move to starting position (top-left corner of the border)
        walls_turtle.penup()
        walls_turtle.goto(self.leftWallPos, self.downWallPos)  # Adjust to be just inside the screen size
        walls_turtle.pendown()

        # # Draw the border (a square or rectangle)
        for i in range(2):
            walls_turtle.forward(self.getWidth())  # Adjust for your board size
            walls_turtle.right(-90)
            walls_turtle.forward(self.getWidth())
            walls_turtle.right(-90)

    def setupPen(self):
        self.pen.speed(0)
        self.pen.shape("square")
        self.pen.color("white")
        self.pen.penup()
        self.pen.hideturtle()
        self.pen.goto(self.centerX, self.upWallPos-self.getHeight()//10)
        self.writeScore(0,0)

    
    def writeScore(self,score,highScore):
        self.pen.clear()
        self.pen.write("Score: {}  High Score: {}".format(score,highScore), align="center", font=("Courier", 24, "normal"))

