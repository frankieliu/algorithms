class Peakable():
    def __init__(self, gen):
        self.gen = gen
        self.get_next()

    def get_next(self):
        try:
            self.next_ = next(self.gen)
            self.stop = False
        except StopIteration:
            self.next_ = None
            self.stop = True

    def peek(self):
        return self.next_

    def __next__(self):
        if self.stop:
            raise StopIteration
        tmp = self.next_
        self.get_next()
        return tmp

    def __iter__(self):
        return self