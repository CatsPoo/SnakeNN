import turtle
import numpy as np

class Snake:
    def __init__(self):
        # Snake head
        self._head = turtle.Turtle()
        self._head.speed(0)
        self._head.shape("square")
        self._head.color("black")
        self._head.penup()
        self._head.goto(0,0)
        self._head.direction = "stop"

        self._segments = []
        self._score = 0
        self._isAlive = True
    
    @property
    def head(self):
        return self._head

    
    @property
    def segments(self):
        return self._segments
    
    @property
    def score(self):
        return self._score
    
    @score.setter
    def score(self,newScore):
        self._score = newScore

    @property
    def isAlive(self):
        return self._isAlive
    
    @isAlive.setter
    def isAlive(self,value):
        self.isAlive = value

    def addSegment(self):
        new_segment = turtle.Turtle()
        new_segment.speed(0)
        new_segment.shape("square")
        new_segment.color("grey")
        new_segment.penup()
        self.segments.append(new_segment)

    def moveSegments(self):
        for index in range(len(self.segments)-1, 0, -1):
            x = self.segments[index-1].xcor()
            y = self.segments[index-1].ycor()
            self.segments[index].goto(x, y)

        # Move segment 0 to where the head is
        if len(self.segments) > 0:
            x = self.head.xcor()
            y = self.head.ycor()
            self.segments[0].goto(x,y)

    def checkForSelfCollition(self):
        for segment in self.segments:
            if segment.distance(self.head) < 20:
                return True
        return False
    
    def clearTail(self):
        # Hide the segments
        for segment in self.segments:
            segment.goto(1000, 1000)
    
        # Clear the segments list
        self.segments.clear()

    def go_up(self):
        if self.head.direction != "down":
            self.head.direction = "up"

    def go_down(self):
        if self.head.direction != "up":
            self.head.direction = "down"

    def go_left(self):
        if self.head.direction != "right":
            self.head.direction = "left"

    def go_right(self):
        if self.head.direction != "left":
            self.head.direction = "right"

    def move(self):
        if self.head.direction == "up":
            y = self.head.ycor()
            self.head.sety(y + 20)

        if self.head.direction == "down":
            y = self.head.ycor()
            self.head.sety(y - 20)

        if self.head.direction == "left":
            x = self.head.xcor()
            self.head.setx(x - 20)

        if self.head.direction == "right":
            x = self.head.xcor()
            self.head.setx(x + 20)
    
    def kill(self):
        self._isAlive = False
        self.clearTail()

    def getAllSegmentsInRow(self,yVal):
        relevantSegments = []
        for segment in self.segments:
            if(segment.ycor() == yVal):
                relevantSegments.append(segment)
        return relevantSegments


    def getAllSegmentsInCol(self,xVal):
        relevantSegments = []
        for segment in self.segments:
            if(segment.xcor() == xVal):
                relevantSegments.append(segment)
        return relevantSegments
