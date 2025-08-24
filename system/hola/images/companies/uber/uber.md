"""
Main issue is figuring out how to set up
streaming in python
"""

"""
    Given a driver position in a 2D space (x,y), a constant number K, and an Stream / Queue of Riders:

    Write a class that continuously consume new riders from that Stream and that is able to offer concurrently an API called GetClosestKRiders().

    K best
    3 new riders come in
    3 log K  
"""
 
from heapq import heappush, heappop

class ComputeRiders:
    def __init__(self, x, y, k, stream):
        self.x = x
        self.y = y
        self.k = k
        self.stream = stream
        self.maxHeap = []
        """
        { id: (x,y) }
        """
   
    def GetClosestRiders(self):
        self.compute()
        # return [(distance to driver, id)...]
        return self.maxHeap
    
    def addRiders(self,new_riders):
        id_,x,y = new_riders
        d = (x-self.x)**2 + (y-self.y)**2
        heappush(self.maxHeap, (d, id_))
        if len(self.maxHeap) > self.k:
            heappop(self.maxHeap)
        
    def compute(self):
        s = next(self.stream)
        print(s)
        self.addRiders(s)

def riderStream():
    while True:
        id_ = 0
        x,y = 1,1
        yield (id_,x,y)
        id_ += 1
        x+=1
        y+=1
        
c = ComputeRiders(0,0,3,riderStream())
for _ in range(10):
    print(c.GetClosestRiders())
    
