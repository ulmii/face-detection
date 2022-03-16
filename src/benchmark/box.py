from shapely.geometry import Polygon

class Box:
    def __init__(self, box_id, x, y, w, h):
        self.box_id = box_id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.poly = Polygon([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    def iou(self, other):
        return self.poly.intersection(other.poly).area / self.poly.union(other.poly).area
    
    def __str__(self):
        return "({}, {}, {}, {})".format(self.x, self.y, self.w, self.h)
    
    def __repr__(self):
        return "Box({}, {}, {}, {})".format(self.x, self.y, self.w, self.h)