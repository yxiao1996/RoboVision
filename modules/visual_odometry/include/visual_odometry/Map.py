class Map():
    def __init__(self):
        self.frames = []

    def pushFrame(self, frame):
        self.frames.append(frame)

    def popFrame(self):
        return self.frames.pop()