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

gameMode = GamkeMode.HUMAN

delay = 0.1
score = 0
high_score = 0

screenPadding = 30

board = Board(600,600)
snake = Snake()
food = Food(board.leftWallPos,board.rightWallPos,board.downWallPos
            ,board.upWallPos)

# Set up the screen
wn = turtle.Screen()
wn.title("Snake Game by @TokyoEdTech")
wn.bgcolor("lime")
wn.setup(width=board.getWidth()+ screenPadding, height=board.getHeight() +screenPadding)
wn.tracer(0) # Turns off the screen updates

# setup board borders
border_turtle = turtle.Turtle()
border_turtle.hideturtle()  # Hide the turtle arrow icon
border_turtle.speed(0)      # Fastest drawing speed
border_turtle.color("black")  # Set the border color to black
border_turtle.pensize(3)    # Set the border thickness

# Move to starting position (top-left corner of the border)
border_turtle.penup()
border_turtle.goto(board.leftWallPos, board.upWallPos)  # Adjust to be just inside the screen size
border_turtle.pendown()

# Draw the border (a square or rectangle)
for i in range(2):
    border_turtle.forward(board.getWidth())  # Adjust for your board size
    border_turtle.right(90)
    border_turtle.forward(board.getHeight())
    border_turtle.right(90)

# Pen
pen = turtle.Turtle()
pen.speed(0)
pen.shape("square")
pen.color("white")
pen.penup()
pen.hideturtle()
pen.goto(0, 260)
pen.write("Score: 0  High Score: 0", align="center", font=("Courier", 24, "normal"))

# comtrols and keyboard settings bindings
if(gameMode == GamkeMode.HUMAN):
    wn.listen()
    wn.onkeypress(snake.go_up, "w")
    wn.onkeypress(snake.go_down, "s")
    wn.onkeypress(snake.go_left, "a")
    wn.onkeypress(snake.go_right, "d")

elif(gameMode == GamkeMode.AI or gameMode == GamkeMode.TRAINING):
    dir = random.randint(0,4)
    if(dir == 0): snake.go_up
    elif(dir == 1): snake.go_down
    elif(dir == 2): snake.go_left
    elif(dir == 3): snake.go_right

# Main game loop
while True:
    wn.update()

    leftWallDistance = np.abs(board.leftWallPos - snake.head.xcor())
    rightWallDistance = np.abs(board.rightWallPos - snake.head.xcor())
    upWallDistance = np.abs(board.upWallPos - snake.head.ycor())
    downWallDistance = np.abs(board.downWallPos - snake.head.ycor())

    nearestWallDistance = np.min([leftWallDistance,rightWallDistance,upWallDistance,downWallDistance])
    nearestWallDistance = np.max([nearestWallDistance,0])

    # Check for a collision with the border
    if (nearestWallDistance == 0):
        time.sleep(1)
        snake.head.goto(0,0)
        snake.head.direction = "stop"
        
        # Clear the segments list
        snake.segments.clear()

        # Reset the score
        score = 0

        # Reset the delay
        delay = 0.1

        pen.clear()
        pen.write("Score: {}  High Score: {}".format(score, high_score), align="center", font=("Courier", 24, "normal")) 


    # Check for a collision with the food
    if snake.head.distance(food()) < 20:
        # Move the food to a random spot
        food.randomLocaation()

        # Add a segment
        snake.addSegment()

        # Shorten the delay
        delay -= 0.001

        # Increase the score
        score += 10

        if score > high_score:
            high_score = score
        
        pen.clear()
        pen.write("Score: {}  High Score: {}".format(score, high_score), align="center", font=("Courier", 24, "normal")) 

    # Move the end segments first in reverse order
    snake.moveSegments()
    snake.move()    

    # Check for head collision with the body segments
    if(snake.checkForSelfCollition()):
        time.sleep(1)
        snake.head.goto(0,0)
        snake.head.direction = "stop"
        snake.clearTail()

        # Reset the score
        score = 0

        # Reset the delay
        delay = 0.1
        
        # Update the score display
        pen.clear()
        pen.write("Score: {}  High Score: {}".format(score, high_score), align="center", font=("Courier", 24, "normal"))

    time.sleep(delay)

wn.mainloop()