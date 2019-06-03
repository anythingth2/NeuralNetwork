import numpy as np
from functools import reduce
count_layer = 0


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


class Input(Layer):
    def __init__(self, shape):
        super().__init__()
        self.output_shape = shape

    def forward(self, x):
        if self.next_layer != None:
            return self.next_layer.forward(x)


class Dense(Layer):

    def __init__(self, size):
        super().__init__()

        self.size = size

    @property
    def number_variable(self):
        return reduce(lambda a, b: a*b, self.weight.shape)

    def compile(self):
        print(f'compiling {self.name}')
        self.weight = np.random.randn(self.size,
                                      self.previous_layer.output_shape[0], )

        self.output_shape = (self.size,)
        if self.next_layer != None:
            self.next_layer.compile()

    def forward(self, x):
        y = np.empty(self.output_shape)
        for index, node in enumerate(y):
            y[index] = np.dot(x, self.weight[index].T)
        return y

    def __str__(self):
        return f'Dense {self.name} size {self.size}  number_variable {self.number_variable}'


model = Input((10,))
model.add_layer(Dense(20))

model.add_layer(Dense(40))

model.compile()
model.summary()

x = np.arange(10)
y = model.forward(x)
y.shape
