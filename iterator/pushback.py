class PushBack():
    def __init__(self, gen):
        self.gen = gen
        self.stack = []

    def __next__(self):
        if len(self.stack) > 0:
            return self.stack.pop()
        return next(self.gen)

    def __iter__(self):
        return self
    
    def push_back(self, element):
        self.stack.append(element)

    