"""二次関数の最小化問題をとく"""

import numpy as np
import chainer

class Quadratic(chainer.Link):
    def __init__(self):
        super().__init__(
            x = (1)
        )
        self.x.data = np.array([2], dtype = np.float64)
    def foward(self):
        return self.x * self.x

model = Quadratic()
optimizer = chainer.optimizers.SGD(lr=0.1) #optimizerの選択と学習率の決定
optimizer.use_cleargrads() #計算の効率化のために入れる(cleargradsを使えるようにしてる）
optimizer.setup(model) #optimizerにモデルをセット

for i in range(80):
    model.cleargrads() #重みを一度に初期化
    y = model.foward()
    y.backward()
    print("=== Epoch %d ===" % (i + 1))
    print("model.x.data = %f" % model.x.data)
    print("y.data = %f" % y.data)
    print("model.x.grad = %f" % model.x.grad)
    print()
    optimizer.update()