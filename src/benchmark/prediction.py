class Prediction:
    def __init__(self, speed, accuracy):
        self.speed = speed
        self.accuracy = accuracy
    
    def stats(self):
        return "Speed: {}ms\n".format(self.speed / 1e+6) + self.accuracy.stats()
    
    def __str__(self):
        return "(Speed: {}ms, Accuracy: {})".format(self.speed / 1e+6, str(self.accuracy))
    
    def __repr__(self):
        return "Prediction({}ms, {})".format(self.speed / 1e+6, repr(self.accuracy))