import numpy as np
from functools import reduce
count_layer = 0

learning_rate = 0.001


class Layer:
    def __init__(self, name=None):
        self.previous_layer = None
        self.next_layer = None
        self.output_shape = None
        if name == None:
            global count_layer
            self.name = f'Layer_{count_layer}'
            count_layer += 1
        else:
            self.name = name

    def compile(self):
        pass

    def forward(self, x):
        pass

    def fit(self, x, y_true):
        pass

    def add_layer(self, layer):
        if self.next_layer != None:
            self.next_layer.add_layer(layer)
        else:
            layer.previous_layer = self
            self.next_layer = layer

    def __str__(self):
        return self.name

    def compile(self):
        print(f'compiling {self.name}')
        # self.compile()
        if self.next_layer != None:
            self.next_layer.compile()

    def summary(self):

        print(str(self))
        if self.next_layer != None:
            self.next_layer.summary()


class Dense(Layer):

    def __init__(self, size):
        super().__init__()
        self.weight = None
        self.size = size

    @property
    def number_variable(self):
        if self.weight is not None:
            return reduce(lambda a, b: a*b, self.weight.shape)
        else:
            return 0

    def compile(self):
        print(f'compiling {self.name}')
        if self.next_layer != None:
            self.next_layer.compile()
            self.weight = np.random.randn(self.next_layer.output_shape[0], self.size,
                                          )
            print(f'weight shape {self.weight.shape}')
            self.output_shape = (self.next_layer.size,)
        else:
            self.output_shape = (self.size,)

    def forward(self, x):
        if self.next_layer == None:
            return x
        y = np.empty(self.output_shape)

        for index, node in enumerate(y):
            y[index] = np.dot(x, self.weight[index].T)

        y = self.next_layer.forward(y)

        return y

    def calculate_loss(self, y_true, y_pred):
        return np.mean((y_true-y_pred)**2)

    def calculate_gradient(self, y_true, y_pred):
        return 2*((y_pred-y_true))/y_true.shape[-1]

    def optimise(self, gradient):
        self.weight = self.weight - learning_rate * gradient

    def fit(self, x, y_true):
        y_pred = self.forward(x)
        loss = self.calculate_loss(y_true, y_pred)
        print(f'loss {loss}')
        gradient = self.calculate_gradient(y_true, y_pred)

        self.optimise(gradient)

    def __str__(self):
        return f'Dense {self.name} size {self.size}  number_variable {self.number_variable}'


size = 20
sample = 1000
x = np.random.rand(size*sample).reshape((-1, size))
y = np.matmul(x, np.ones((x.shape[1], 1)))
shuffle_index = np.arange(sample)
np.random.shuffle(shuffle_index)

x = x[shuffle_index]
y = y[shuffle_index]

model = Dense(size)
model.add_layer(Dense(1))

model.compile()
model.summary()




for index in range(len(x)):
    sample = x[index]
    y_true = y[index]
    model.fit(sample, y_true)


model.forward(np.ones(20)/10)
