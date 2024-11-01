# Simple Snake Game in Python 3 for Beginners
# By @TokyoEdTech

import turtle
import time
import random
import numpy as np
from objects.board import Board
from objects.snake import Snake
from objects.food import Food

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

# Functions

def go_up():
    if snake.head.direction != "down":
        snake.head.direction = "up"

def go_down():
    if snake.head.direction != "up":
        snake.head.direction = "down"

def go_left():
    if snake.head.direction != "right":
        snake.head.direction = "left"

def go_right():
    if snake.head.direction != "left":
        snake.head.direction = "right"

def move():
    if snake.head.direction == "up":
        y = snake.head.ycor()
        snake.head.sety(y + 20)

    if snake.head.direction == "down":
        y = snake.head.ycor()
        snake.head.sety(y - 20)

    if snake.head.direction == "left":
        x = snake.head.xcor()
        snake.head.setx(x - 20)

    if snake.head.direction == "right":
        x = snake.head.xcor()
        snake.head.setx(x + 20)

# Keyboard bindings
wn.listen()
wn.onkeypress(go_up, "w")
wn.onkeypress(go_down, "s")
wn.onkeypress(go_left, "a")
wn.onkeypress(go_right, "d")

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

        # Hide the segments
        for segment in snake.segments:
            segment.goto(1000, 1000)
        
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
    move()    

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