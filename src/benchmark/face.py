class Face:
    def __init__(self, face_id, box):
        self.face_id = face_id
        self.box = box
        
    def __eq__(self, other):
        return self.face_id == other.face_id
    
    def __str__(self):
        return "({}, {})".format(self.face_id, str(self.box))
    
    def __repr__(self):
        return "Face({}, {})".format(self.face_id, repr(self.box))