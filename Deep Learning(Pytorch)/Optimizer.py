import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
import matplotlib.pyplot as plt

# hyper parameters
LR = 0.01
BATCH_SIZE = 20
EPOCH = 20

# test data
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

# plt.scatter(x.numpy(), y.numpy())
# plt.show()

torch_dataset = torch.utils.data.TensorDataset(x, y)
data_loader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(1, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = func.relu(self.hidden(x))
        x = self.output(x)
        return x


net_SGD = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.9)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = nn.MSELoss()
loss_history = [[], [], [], []]

for epoch in range(EPOCH):
    print(epoch)
    for step, (batch_x, batch_y) in enumerate(data_loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        for net, opt, l_history in zip(nets, optimizers, loss_history):
            out = net(b_x)
            loss = loss_func(out, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_history.append(loss.data)


labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_history in enumerate(loss_history):
    plt.plot(l_history, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()