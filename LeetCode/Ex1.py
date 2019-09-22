def ask_ok (prompt, retries = 4, reminder = "Please Try AGain!"):
    while True:
        ok = input(prompt)
        if ok in ('y', 'ye', 'yes' ):
            return True
        if ok in ('n', 'no', 'nope', 'nope'):
            return False
        retries = retries -1
        if retries < 0:
            raise ValueError('invalid user response')
        print (reminder)


def parrot(voltage, state = 'a stiff', action = 'voom', type = 'Norwegian Blue'):
    print("-- This parrot wouldn't", action, end=' ')
    print("if you put", voltage, "volts through it.")
    print("-- Lovely plumage, the", type)
    print("-- It's", state, "!")


parrot(voltage = 120)

fruits = ["Apple", "Pears", "Peach", "Berries", "Apple"]
fruits.count("Apple")

fruits.index("Apple", 3)
fruits.append("Grape")

fruits.pop()

from collections import deque 

queue = deque(fruits)
queue.popleft()

squares = [x**2 for x in range (10)]

combs = [(x,y) for x in [1,3,5,6,7] for y in [1,4,5] if x!=y]
combs

vec = [4,-2,5,7,0]
[x*2 for x in vec]
[x for x in vec if x >= 0]
[abs(x) for x in vec]


spacedfruit = []
for fruit in fruits:
    spacedfruit.append("   "+ fruit+ "   ")

spacedfruit = ["  "+fruit+"   "for fruit in fruits]

[weapon.strip() for weapon in spacedfruit]

[(x,x**2) for x in range(5)]

matrix = [fruits,vec]
flatmatrix = [second for array in matrix for second in array]
flatmatrix


from math import pi 
[str(round(pi,i)) for i in range(1,9)]


[row for row in matrix]
[[row[i] for row in matrix] for i in range(len(matrix[0]))]

sub = [i for i in range (5)]
matrix = [sub for n in range (5)]
matrix

list('abc')
list("123")

list(zip(*matrix))

a = 2,5,6,5,6
b,c,d,e,f = a

t = '50',[23,3,6],a
set(fruits)
{"Apples", "ApPles","Pears","Berries"}

"Apple" in fruits

a = set("abracadabra")
b = set('alacazam')
a-b
a|b
a&b
a^b

a = [x for x in "abracadabra" if x not in "abc"]
set(a)
a = {x for x in "abracadabra" if x not in "abc"}
a

pouf = {"whoops":"late", "hello":"early"}
pouf["hello"]
pouf["yup"] = "on time"
del pouf["yup"]
pouf

for k,v in pouf.items():
    print (k,v)

for i, n in enumerate([4,2,6]):
    print (i,n)

{x: x**2 for x in range (2,7)}
import fibo

fibo.fib(9) 

knights = {'gallahad': 'the pure', 'robin':'the brave'}
for k,d in knights.items():
    print (d)


words = ['cat', 'window', 'defenestrate']

for en,w in enumerate (words[:]):
    if len(w) > 4:
        words.insert(en-1,w)


f = open ('workfile', 'rb+')
f.write()

while True:
    try:
        x = int(input("Please enter a number:"))
        break
    except ValueError:
        print ("Try again:")


import sys 

try:
    f = open('myfile.txt')
    s = f.readline()
    i = int(s.strip())
except IOError as err:
    print ("I/O error: {0}".format(err))
except ValueError:
    print ("Could not convert data to an interger")
except:
    print ("Unexpected error:", sys.exc_info()[0])
    raise


for arg in sys.argv[1:]:
    try:
        f= open(arg, 'r')
    except IOError:
        print ("Cannot open", arg)
    else:
        print(arg, 'has', len(f.readlines()),'lines')
        f.close()

try:
    raise NameError("Hi There")
except NameError:
    print ('An exception flew by!')
    raise 





class MyError(Exception):
    def __init__


import numpy as np
a = np.concatenate(([3], [0]*5, np.arange(-1, 1.002, 2/9.0)))


np.arange(0,30.1,5)
np.linspace(-5,10)