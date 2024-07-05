我这个代码在准确率大约到达50%的时候遇到了局部最优问题，不论lr设置为0.01还是0.001，都没有解决。

```
    def compute_gradient(self, criterion: nn.CrossEntropyLoss, device: str, num_epochs: int):
        self.model.to(device)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        
        total_loss = 0.0
        for epoch in range(num_epochs):
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
        return gradient_changes, total_loss / (len(self.data_loader) * num_epochs)
```