import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
    '/files/', train=True, download=True, transform=torchvision.transforms.ToTensor()))
test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
    '/files/', train=False, download=True, transform=torchvision.transforms.ToTensor()))


class rnn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(rnn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out


model = rnn(input_size, hidden_size, num_layers, num_classes).to(device)
crit = nn.CrossEntropyLoss()
optimizers = optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (dat, target) in enumerate(train_loader):
        dat = dat.reshape(-1, sequence_length, input_size).to(device)
        target = target.to(device)
        output = model(dat)
        loss = crit(output, target)
        optimizers.zero_grad()
        loss.backward()
        optimizers.step()
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for dat, target in enumerate(test_loader):
        dat = dat.reshape(-1, sequence_length, input_size).to(device)
        target = target.to(device)
        output = model(dat)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(
        100 * correct / total))
torch.save(model.state_dict(), 'C:/users/joyje/OneDrive/Desktop/rnnmodel.pth')
