from game import Game
import turtle
from gameMode import GamkeMode
from solver import Solver
import concurrent.futures


boardWidth = 400
boardHeight = 400
boaedOffsst = 20

gridRows = 1
gridCols = 3

def main():
    screen = setupScreen(gridRows,gridCols)
    population = generatePopulation(2)
    
    for  instanceNum,solver in enumerate(population):
        fitnessFunction(solver,screen=screen,numOfEpisodes=5,instanceNumber=instanceNum)

    pass

def setupScreen(rows,cols):
    screen = turtle.Screen()
    screen.title("Snake Game by @TokyoEdTech")
    screen.bgcolor("lime")
    screen.tracer(0) # Turns off the screen updates

    screenWidth = boaedOffsst + (boardWidth * cols)
    screenHeight = boaedOffsst + (boardHeight * rows)

    screen.setup(width=screenWidth, height=screenHeight)
    screen.setworldcoordinates(0, 0, screenWidth, screenHeight)

    # Create a turtle for drawing grid lines
    grid_drawer = turtle.Turtle()
    grid_drawer.speed(0)  # Set the drawing speed to max
    grid_drawer.hideturtle()

    # Draw vertical grid lines
    for col in range(1, cols):
        x = col * boardWidth
        grid_drawer.penup()
        grid_drawer.goto(x, 0)
        grid_drawer.pendown()
        grid_drawer.goto(x, screenHeight)

    # Draw horizontal grid lines
    for row in range(1, rows):
        y = row * boardHeight
        grid_drawer.penup()
        grid_drawer.goto(0, y)
        grid_drawer.pendown()
        grid_drawer.goto(screenWidth, y)

    return screen

def generatePopulation(num):
    return [Solver() for _ in range(num)]

def fitnessFunction(network,screen,numOfEpisodes=5,instanceNumber = 0):
    game = None 
    scores = 0
    for episode in range(numOfEpisodes):
        game = Game(GamkeMode.AI,wn=screen,solver=network,boardWidth= boardWidth,boardHeight=boardHeight,boardOffset = boaedOffsst//2,row=instanceNumber//gridCols,col=instanceNumber%gridCols)
        game.runGame()
        print('DIE!!!!')
        scores+=game.score
        game.endGame()
        game=None
    return scores/numOfEpisodes

if(__name__ == '__main__'):
    main()