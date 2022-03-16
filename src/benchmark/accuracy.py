class Accuracy:
    def __init__(self, ious, positives, false_positives, negatives):
        self.ious = ious
        self.positives = positives
        self.false_positives = false_positives
        self.negatives = negatives
        
        self.precision = positives / (positives + false_positives) if positives + false_positives > 0 else 0 
        self.recall = positives / (positives + negatives)
        
    def stats(self):
        return "Precision: {}".format(self.precision) \
            + "\nRecall: {}".format(self.recall) \
            + "\nIous: {}".format(str(self.ious)) \
            + "\nPositives: {}, False Positives: {}, Negatives: {}".format(self.positives, self.false_positives, self.negatives)
        
    def __str__(self):
        return "(Ious: {}, Positives: {}, False Positives: {}, Negatives: {})".format(str(self.ious), self.positives, self.false_positives, self.negatives)
    
    def __repr__(self):
        return "Prediction({}, {}, {}, {})".format(repr(self.ious), self.positives, self.false_positives, self.negatives)
