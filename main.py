import torch


class MyClass:
    def __init__(self):
        pass

    def __call__(self):
        return 3

    def ciao(self):
        print(3)

    def ciao(self):
        print(6)


class MySubclass(MyClass):
    def forward():
        return 'hello'


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(100, 100)

    def forward(self, x):
        return self.lin1(x)


def train():
    pass


def test():
    pass


def main():
    model = MyModel()
    loss_fn = None
    x = torch.rand(100)
    y = model(x)
    loss = loss_fn(y, y_giusta)
    loss_fn.backward()


if __name__ == "__main__":
    main()
