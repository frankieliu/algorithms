class Peakable:
    
    def __init__(self, iter_):
        self.iter = iter_ 
        self.get_next_()

    def __next__(self):
        if self.stop:
            raise StopIteration
        tmp = self.next_
        self.get_next_()
        return tmp

    def get_next_(self):
        try:
            self.next_ = next(self.iter)
            self.stop = False
        except StopIteration:
            self.next_ = None
            self.stop = True
        
    def peek(self):
        if self.next_:
            return self.next_
        self.get_next_()
        return self.next_

    def __iter__(self):
        return self