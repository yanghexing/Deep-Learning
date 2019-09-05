import torch
from torch.autograd import Variable
import torch.nn.functional as func
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 200), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = func.relu(self.hidden(x))
        x = self.output(x)
        return x


net = Net(1, 10, 1)
plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

for i in range(100):
    prediction = net(x)
    loss = loss_func(prediction, y)
    if i % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=1)
        plt.pause(0.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.ioff()
plt.show()