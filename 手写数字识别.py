import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from dataset.mnist import load_mnist


class SGD:
    """随机梯度下降法（用于更新参数）"""
    def __init__(self, lr = 0.01):
        self.lr = lr    #设计学习率

    def update(self, params, grads):
        """设计参数更新方式"""
        #for key in params.keys():
        for key in ('W1', 'b1', 'W2', 'b2'):
            params[key] -= self.lr * grads[key]


class ReLU:
    """ReLU函数层
    当x<=0时，y=0，当x>0时，y=x"""
    def __init__(self):
        self.mask = None    #设计一个实例变量

    def forward(self, x):
        self.mask = (x <= 0)   #当x小于等于0时保存为True,大于0时保存为False
        out = x.copy()       #复制x赋值为out
        out[self.mask] = 0    #将True的位置赋值为0，其他位置不变

        return out

    def backward(self, dout):
        dout[self.mask] = 0   #原理同上
        dx = dout

        return dx


class Affine:
    """仿射变换，进行的矩阵的乘积运算，几何中，仿射变换包括一次线性变换和一次平移，分别对应神经网络的加权和运算与加偏置运算"""
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """正向传播，y=np.dot(x,w)+b"""
        # 对应张量
        """将输入数组 x 的原始形状存储起来，并将其重新调整为二维的形状，以方便后续的矩阵计算"""
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)

        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        """反向传播，梯度乘上翻转值的转置矩阵"""
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)   #当axis为0时,是压缩行,即将每一列的元素相加,将矩阵压缩为一行

        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）

        return dx


class Softmax:
    """处理较灵活，能够处理1，2维数组"""
    def __init__(self, x):
        self.x = x

    def forward(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x - np.max(x) # 溢出对策
        return np.exp(x) / np.sum(np.exp(x))


class SoftmaxWithLoss:
    """使用交叉熵误差为Softmax函数计算损失函数"""
    def __init__(self):
        self.loss = None   #损失
        self.y = None    #Softmax函数的输出
        self.t = None   #监督数据，使用one-hot方式编码

    def cross_entropy_error(self, y, t):
        """交叉熵误差"""
        if y.ndim ==1:   #ndim返回的是数组的维度，返回的只有一个数，该数即表示数组的维度
            t = t.reshape(1, t.size)   #将t矩阵转换为一个1维数组
            y = y.reshape(1, y.size)   #同上

        """处理 t（目标标签）和 y（模型预测值）的形状不一致的情况
        if t.size == y.size:这一条件语句检查t和y的元素总数是否相等。如果它们的size相等，说明t是一个one-hot编码向量。
        t = t.argmax(axis=1):使用argmax(axis=1)会将每一行中最大值的位置作为该样本的类别索引。
        这行代码的目的是从 one-hot编码转换成类别索引，这样t就可以变成一个一维数组，每个元素代表对应样本的类别。"""
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]
        error = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

        return error

    def forward(self, x, t):
        self.t = t
        softmax = Softmax(x)
        self.y = softmax.forward(x)
        self.loss = self.cross_entropy_error(self.y, self.t)     #使用交叉熵误差计算loss

        return self.loss

    def backward(self, dout = 1):
        """使用计算值减去监督值再除以批次的大小即为单个数据的误差"""
        batch_size = self.t.shape[0]   #batch_size表示批次的大小
        dx = (self.y - self.t) / batch_size   #除以batch_size后，传递给前面的层的是单个数据的误差

        return dx


class TwoLayerNet:
    """含有两个隐藏层的神经网络"""
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        """weight_init_std为初始化权重时的高斯分布规模"""

        #初始化权重
        self.params = {}   #保存神经网络参数的字典型变量
        """np.random.randn()生成一个服从标准正态分布（均值为 0，标准差为 1）的随机数或随机数组
        np.zeros用来创建一个指定形状的数组，并用全零填充"""
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        #生成层
        self.layers = OrderedDict()      #保存神经网络层的有序字典型变量
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()      #神经网络的最后一层

    def predict(self, x):
        """将值代入神经网络的正向传播，所得结果即为神经网络的预测值，参数x是图像数据"""
        for layer in self.layers.values():   #遍历所有层级，使传入值经过神经网络所有层级的处理
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """x为图像数据，t为正确解标签"""
        y = self.predict(x)
        loss = self.lastLayer.forward(y, t)    #使用SoftmaxWithLoss计算loss
        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)        #用来返回数组中最大数值的索引，注意其是从0开始
        if t.ndim != 1 : t = np.argmax(t, axis=1)     #ndim返回的是数组的维度，返回的只有一个数，该数即表示数组的维度

        """y == t:这是一个逐元素比较操作，y == t 会生成一个布尔数组，其中 True 表示预测正确，False 表示预测错误。
        np.sum(y == t):将布尔数组转换为整数（True 转为 1，False 转为 0），然后求和。这个求和结果就是预测正确的样本数量。
        x.shape[0]：取结果的第一个维度。"""
        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    def numerical_gradients(self, x, t):
        """使用数值微分法求梯度"""
        loss_W = lambda W: self.loss(x, t)    #匿名函数，将函数式子传递给数值微分

        def numerical_gradient(f, x):
            """定义数值微分法"""
            h = 1e-4  # 0.0001
            grad = np.zeros_like(x)    #np.zeros_like（）创建一个与指定数组具有相同形状和数据类型的新数组，并用全零填充

            it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                tmp_val = x[idx]
                x[idx] = float(tmp_val) + h
                fxh1 = f(x)  # f(x+h)

                x[idx] = tmp_val - h
                fxh2 = f(x)  # f(x-h)
                grad[idx] = (fxh1 - fxh2) / (2 * h)

                x[idx] = tmp_val  # 还原值
                it.iternext()

            return grad

        grads = {}
        #存储各个参数的梯度
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        """使用误差反向传播计算梯度"""
        #forward
        self.loss(x, t)

        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        #设定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 500
learning_rate = 0.1
epoch = 1
train_loss_list = []
train_acc_list = []
test_acc_list = []
epoch_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #通过误差反向传播法求梯度
    grad = network.gradient(x_batch, t_batch)

    #参数更新
    sgd = SGD(learning_rate)
    sgd.update(network.params, grad)

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f'第{epoch}次训练精度为：')
        epoch_list.append(epoch)
        epoch = epoch + 1
        print(train_acc, test_acc)


plt.plot(epoch_list, train_acc_list, label='train_accuracy')
plt.plot(epoch_list, test_acc_list, label='test_accuracy')
plt.title('train_acc & test_acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

