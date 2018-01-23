
class point():
    def __init__(self):
        self.a = 0
        self.b = 0
    
    def setPos(self, a, b):
        self.a = a
        self.b = b

class rectengle():
    def __init__(self, x, y, w, h):
        self.RT = point()
        self.LB = point()
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
        self.setCoor()
    
    def setCoor(self):
        rt_x = self.x + self.w/2
        rt_y = self.y - self.h/2
        lb_x = self.x - self.w/2
        lb_y = self.y + self.h/2
        self.RT.setPos(rt_x, rt_y)
        self.LB.setPos(lb_x, lb_y)

def IOU(A, B):
    W = min(A.RT.a, B.RT.a) - max(A.LB.a, B.LB.a)
    H = min(A.LB.b, B.LB.b) - max(A.RT.b, B.RT.b)
    if W <= 0 or H <= 0:
        return 0
    SA = (A.RT.a - A.LB.a) * (A.LB.b - A.RT.b)
    SB = (B.RT.a - B.LB.a) * (B.LB.b - B.RT.b)
    cross = W * H
    return cross/(SA + SB - cross)

def CalcuIoU(x1, y1, w1, h1, x2, y2, w2, h2):
    rect1 = rectengle(x1, y1, w1, h1)
    #rect1.setCoor(x1, y1, w1, h1)
    rect2 = rectengle(x2, y2, w2, h2)
    #rect2.setCoor(x2, y2, w2, h2)
    iou = IOU(rect1, rect2)

    return iou