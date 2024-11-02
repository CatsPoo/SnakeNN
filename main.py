from game import Game
import turtle
from gameMode import GamkeMode

def main():
    screen = turtle.Screen()
    g = Game(gameMode=GamkeMode.AI,wn=screen)
    g.runGame()

if(__name__ == '__main__'):
    main()