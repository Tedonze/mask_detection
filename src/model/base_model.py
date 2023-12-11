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
            number_item = 0
            for data in dataloader[0]:
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                number_item += 1
            loss_training = running_loss/number_item
            print(f'''epoch: {epoch},.Loss:{loss_training :.3f}''')
            running_loss = 0.0
            number_item = 0
            for  data  in dataloader[1]:
                    inputs, labels = data
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item()
                    number_item += 1
            loss_validation = running_loss/number_item
            print(f'''epoch: {epoch},.Loss:{loss_validation :.3f}''')

                    

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_dict(path)

    
        








        


