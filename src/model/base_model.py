import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from src.dataloader.loader import IMG_SIZE
from src.utils import get_device

device = get_device()


class EarlyStopping:
    def __init__(self, diff, max_try) -> None:
        self.count = 0
        self.best_validation_loss = float('inf')
        self.diff = diff
        self.max_try = max_try
        self.save = False

    def stop_ilteration(self, validation_loss):
        if validation_loss < self.best_validation_loss + self.diff:
            self.count = 0
            self.best_validation_loss = validation_loss
            self.save = True
        else:
            self.save = False
            self.count = self.count + 1
        if self.count > self.max_try:
            print("break")
        return self.count > self.max_try


class BaseModel:
    def __init__(self, model, lr=1e-3, momentum=0.9):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum
            )

    def fit(self, dataloader, n_epoch=100,  diff=1e-4, max_try=10):
        """_summary_

        Args:
            dataloader (_type_): _description_
            n_epoch (int, optional): _description_. Defaults to 10.
            diff (_type_, optional): _description_. Defaults to 1e-4.
            max_try (int, optional): _description_. Defaults to 100.
        """
        self.model = self.model.to(device)
        print(device)
        for epoch in range(n_epoch):
            running_loss = 0.0
            number_item = 0
            for data in dataloader[0]:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                number_item += 1
            loss_training = running_loss/number_item
            print(f'''epoch: {epoch},.Loss training:{loss_training :.3f}''')
            running_loss = 0.0
            number_item = 0
            for data in dataloader[1]:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                number_item += 1
            loss_validation = running_loss/number_item
            print(f'''epoch: {epoch},
            .Loss validation:{loss_validation :.3f}''')
            early_stop = EarlyStopping(diff, max_try)
            if early_stop.stop_ilteration(loss_validation):
                break
            if early_stop.save:
                self.save_model(f"loss_validation{loss_validation:.3f}.save")       
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def predict(self, image):
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image = ToTensor()(image)
        image = image.to(device)
        outputs = self.model(image)
        return outputs.numpy()


    



             
              
         

        
              
         
             








        


