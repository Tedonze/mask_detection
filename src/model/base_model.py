import torch
import torch.nn as nn
import torch.optim as optim


class BaseModel():
    def __init__(self, model, lr=1e-3, momentum=0.9):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.parameters(), 
            lr=lr,
            momentum=momentum
            )

    def fit(self, dataloader, n_epoch=10):
        for epoch in range(n_epoch):
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 10 == 9:
                    print(f'''epoch: {epoch},batch number {i+1} 
                    .Loss:{running_loss/10 :.3f}''')
                    running_loss = 0

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_dict(path)

    
        








        


