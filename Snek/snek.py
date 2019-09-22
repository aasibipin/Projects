import math
import random
import tkinter as tk
from tkinter import messagebox
import pygame

class cube(object):
    rows = 20
    w = 500

    def __init__ (self, start, dirnx = 1, dirny = 0, color = (255, 0, 0)):
        self.position = start
        self.dirnx = 1
        self.dirny = 0 
        self.color = color

    def move(self, dirnx, dirny):
        self.dinrx = dirnx
        self.dirny = dirny
        self.position = (self.position[0] + self.dirnx, self.position[1] + self.dirny)
 
    def draw(self, surface, eyes = False):
        dis = self.w//self.rows
        i = self.position[0]
        j = self.position[1]

        pygame.draw.rect(surface, self.color, (i*dis+1,j*dis+1, dis-2, dis-2))
        if eyes:
            centre = dis//2
            radius = 3
            circleMiddle = (i*dis+centre-radius,j*dis+8)
            circleMiddle2 = (i*dis + dis -radius*2, j*dis+8)
            pygame.draw.circle(surface, (0,0,0), circleMiddle, radius)
            pygame.draw.circle(surface, (0,0,0), circleMiddle2, radius)
        

class snake():
    body = []
    turns = {}              #Contain dict of the cordinate of turn and the direction 
    
    def __init__ (self, color, position):
        self.color = color
        self.head = cube(position)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
   
    def move(self):
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT:
                pygame.quit()
            
            keys = pygame.key.get_pressed()

            for key in keys:
                if keys[pygame.K_LEFT]:
                    self.dirnx = -1
                    self.dirny = 0
                    self.turns[self.head.position[:]] = [self.dirnx, self.dirny]     # ?????? 
                
                elif keys[pygame.K_RIGHT]:
                    self.dirnx = 1
                    self.dirny = 0
                    self.turns[self.head.position[:]] = [self.dirnx, self.dirny]     # ?????? 

                elif keys[pygame.K_UP]:
                    self.dirnx = 0
                    self.dirny = -1
                    self.turns[self.head.position[:]] = [self.dirnx, self.dirny]     # ?????? 

                elif keys[pygame.K_DOWN]:
                    self.dirnx = 0
                    self.dirny = 1
                    self.turns[self.head.position[:]] = [self.dirnx, self.dirny]     # ?????? 
        
        for i, c in enumerate (self.body):
            p = c.position[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body)-1:
                    self.turns.pop(p)
                    
            else:               #When snake passes the borders
                if c.dirnx == -1 and c.position[0] <= 0: c.position = (c.rows-1, c.position[1])
                elif c.dirnx == 1 and c.position[0] >= c.rows-1: c.position = (0,c.position[1])
                elif c.dirny == 1 and c.position[1] >= c.rows-1: c.position = (c.position[0], 0)
                elif c.dirny == -1 and c.position[1] <= 0: c.position = (c.position[0],c.rows-1)
                else: c.move(c.dirnx,c.dirny)

    def reset(self, position):
        self.head = cube(position)
        self.body = []
        self.body.append(self.head)
        self.turns={}
        self.dirnx = 0
        self.dirny = 1


    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny
        
        if dx == 1 and dy == 0:
            self.body.append(cube((tail.position[0]-1, tail.position[1])))
        elif dx == -1 and dy == 0:
            self.body.append(cube((tail.position[0]+1, tail.position[1])))
        elif dx == 0 and dy == 1:
            self.body.append(cube((tail.position[0], tail.position[1]-1)))
        elif dx == 0 and dy == -1:
            self.body.append(cube((tail.position[0], tail.position[1])))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy
        
    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i ==0:
                c.draw(surface, True)
            else:
                c.draw(surface)


def drawGrid (w, rows, surface):
    sizeBtwn = w // rows

    x = 0
    y = 0
    for l in range(rows):
        x = x + sizeBtwn
        y = y + sizeBtwn

        pygame.draw.line(surface, (255,255,255), (x,0), (x,w))
        pygame.draw.line(surface, (255,255,255), (0,y), (w,y))

def redrawWindow(surface):
    global rows, width, viper, biscuit
    surface.fill((0,0,0))
    viper.draw(surface)
    drawGrid(width, rows, surface)
    pygame.display.update()

def randomSnack (rows, item):

    positions = item.body

    while True:
        x = random.randrange(rows)
        y = random.randrange(rows)
        if len(list(filter(lambda z:z.position == (x,y), positions))) > 0:
            continue
        else:
            break

    return (x,y)

def message_box(subject, content):
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    messagebox.showinfo(subject, content)
    try:
        root.destroy()
    except:
        pass



def main():
    global width, rows, viper, biscuit
    width = 500
    rows = 20
    win = pygame.display.set_mode((width, width))
    viper = snake((255, 0 , 0), (10, 10))
    biscuit = cube(randomSnack(rows, viper), color = (0,255,0))

    flag = True

    clock = pygame.time.Clock()             #create an object to help track time

    while flag:
        pygame.time.delay(50)               #pause the program for an amount of time
        clock.tick (10)                     #keep the game running slower than 10 ticks per second
        viper.move()

        if viper.body[0].position == biscuit.position:        #If head of viper hits snack
            viper.addCube()
            biscuit = cube(randomSnack(rows, viper), color = (0,255,0))

        for x in range (len(viper.body)):
            if viper.body[x].position in list(map(lambda z:z.position,viper.body [x+1:])):
                print ('Score: ', len (viper.body))
                message_box('Vipers do not eat themselves.', 'Try again.')
                viper.reset((10,10))
                break
        
        redrawWindow(win)
main()           