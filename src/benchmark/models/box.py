#! /usr/bin/env python3

from shapely.geometry import Polygon

class Box:
    def __init__(self, **kwargs):
        if 'box_id' not in kwargs:
            raise ValueError("box_id cannot be empty")

        if 'x1' not in kwargs or 'y1' not in kwargs:
            raise ValueError("Box positions of x1 and y1 must be specified")

        self.box_id = kwargs['box_id']

        self.x1 = kwargs['x1']
        self.y1 = kwargs['y1']

        if 'x2' in kwargs and 'y2' in kwargs:
            self.x2 = kwargs['x2']
            self.y2 = kwargs['y2']

            self.poly = Polygon([[self.x1, self.y1], [self.x2, self.y1], [self.x2, self.y2], [self.x1, self.y2]])
        elif 'w' in kwargs and 'h' in kwargs:
            w = kwargs['w']
            h = kwargs['h']

            self.x2 = self.x1 + w
            self.y2 = self.y1 + h

            self.poly = Polygon([[self.x1, self.y1], [self.x2, self.y1], [self.x2, self.y2], [self.x1, self.y2]])
        else:
            raise ValueError("Either [x2, y2] or [w, h] must be specified")

    def iou(self, other):
        return self.poly.intersection(other.poly).area / self.poly.union(other.poly).area
    
    def __str__(self):
        return "({}, {}, {}, {})".format(self.x1, self.y1, self.x2, self.y2)
    
    def __repr__(self):
        return "Box({}, {}, {}, {})".format(self.x1, self.y1, self.x2, self.y2)