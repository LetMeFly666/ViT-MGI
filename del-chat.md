我当前客户端的代码如下：

```
# 客户端类
class Client:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.model: Optional[CustomViTModel] = None
    
    def set_model(self, model: CustomViTModel, device: str, name: str=None):
        self.model = model
        self.model.to(device)
        self.model.setName(name)
        self.initial_state_dict = copy.deepcopy(self.model.state_dict())
    
    def compute_gradient(self, criterion: nn.CrossEntropyLoss, device: str):
        self.model.to(device)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        total_loss = 0.0
        for images, labels in self.data_loader:
            optimizer.zero_grad()  # 每个批次前清零梯度
            images, labels = images.to(device), labels.to(device)
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()  # 计算当前批次的梯度
            total_loss += loss.item()
            optimizer.step()
        
        # 计算梯度变化
        final_state_dict = self.model.state_dict()
        gradient_changes = {}
        for key in self.initial_state_dict:
            gradient_changes[key] = final_state_dict[key] - self.initial_state_dict[key]
        return gradient_changes, total_loss / len(self.data_loader)

    def getName(self) -> str:
        return self.model.getName()
```

我想让客户端每次训练3个epoch，请你帮我修改`Client`类中的`compute_gradient`函数。

注意，你只需要返回`Client`类相关的代码，不要返回多余的代码。