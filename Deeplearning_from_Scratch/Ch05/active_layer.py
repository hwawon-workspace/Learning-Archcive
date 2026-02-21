class ReLU:
        def __init__(self):
            self.mask = None # 변수 초기화
        
        def forwrad(self, x):
            self.mask = (x <= 0)
            out = x.copy()
            out[self.mask] = 0 # mask에 해당하는 부분은 0으로 채움
            return out
        
        def backward(self, dout):
            dout[self.mask] = 0
            dx = dout

class Sigmoid():
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx