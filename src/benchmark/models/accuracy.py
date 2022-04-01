#! /usr/bin/env python3

class Accuracy:
    def __init__(self, ious, positives, false_positives, negatives):
        self.ious = ious
        self.positives = positives
        self.false_positives = false_positives
        self.negatives = negatives
        
        self.precision = positives / (positives + false_positives) if positives + false_positives > 0 else 0.0 
        self.recall = positives / (positives + negatives) if positives + negatives > 0 else 0.0

        self.f1_score = (self.precision * self.recall) / ((self.precision + self.recall) / 2) if self.precision + self.recall > 0 else 0.0
        
    def stats(self):
        return "Precision: {}".format(self.precision) \
            + "\nRecall: {}".format(self.recall) \
            + "\nF1 Score: {}".format(self.f1_score) \
            + "\nIous: {}".format(str(self.ious)) \
            + "\nPositives: {}, False Positives: {}, Negatives: {}".format(self.positives, self.false_positives, self.negatives)
        
    def write(self):
        return [self.precision, self.recall, self.f1_score, self.ious, self.positives, self.false_positives, self.negatives]

    def __str__(self):
        return "(Ious: {}, Positives: {}, False Positives: {}, Negatives: {})".format(str(self.ious), self.positives, self.false_positives, self.negatives)
    
    def __repr__(self):
        return "Prediction({}, {}, {}, {})".format(repr(self.ious), self.positives, self.false_positives, self.negatives)
