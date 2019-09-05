import torch
from torch.autograd import Variable
import torch.nn.functional as func
import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data,1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1),).type(torch.LongTensor)

x, y = Variable(x), Variable(y)
# plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c=y.data.numpy(), s=100, lw=0)
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


net = Net(2, 10, 2)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
loss_fc = torch.nn.CrossEntropyLoss()

plt.ion()
plt.show()

for i in range(100):
    out = net(x)
    loss = loss_fc(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 2 == 0:
        plt.cla()
        prediction = torch.max(func.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, lw=0, s=100)
        accuracy = sum(pred_y==target_y)/200
        plt.text(1.5,-4,'accuracy=%.2f'%accuracy)
        plt.pause(0.1)


plt.ioff()
plt.show()


