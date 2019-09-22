def scope_test():
    def do_local():
        spam = "local spam"
        print (spam)
    def do_nonlocal():
        nonlocal spam
        spam = "nonlocal spam"
    def do_global():
        global spam
        spam = "global spam"
    spam = "test spam"
    do_local()
    print ("After Local assignment:", spam)
    do_nonlocal()
    print ("After nonlocal assignment:", spam)
    do_global()
    print ("After global assigment:", spam)

scope_test()
print ("In global scope", spam)

class ClassName:
    """A Simple Example Class"""
    i = 12345
    def f(self):
        return "HELLO WORLD"

ClassName.i
ClassName.f
x = ClassName()
x.__doc__

class Complex:
    def __init__ (self, realpart, imagpart) -> "str":
        self.r = realpart
        self.i = imagpart

x = Complex(3.0,-4.5)

Complex.__init__.__annotations__

x.r + x.i

x.counter = 1
while x.counter < 10:
    x.counter = x.counter *2
print (x.counter) 
del x.counter 

x = ClassName()
x.f()

def check (jewls, stones):
    return [s in jewls for s in stones]


def check (jewls, stones):
    return len([s for s in stones if s in jewls])


check ("ab", "baaa")

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

x = [4,6,2,7,1]
y = TreeNode(x)
y.val
def rangsum (root: TreeNode, L: int, R: int) -> int:

    #return root.val
    return sum([v for v in root.val if v in range (L,R)])

rangsum (y,2,5)
