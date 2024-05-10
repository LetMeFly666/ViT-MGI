import torch
import timm


class ModelViT():
    def __init__(self, model_name="vit_base_patch16_224", pretrained=False, num_classes=10, device="cpu"):
        self.epoch = 50
        self.learning_rate = 0.005
        self.device = device
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes).to(self.device)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def forward(self, x):
        return self.model(x)

    def train(self, X: torch.Tensor, y: torch.Tensor, batch_size: int, local_epoch: int, revert=False):
        param = self.get_flatten_parameters()
        loss = 0
        acc = 0
        for epoch in range(local_epoch):
            batch_idx = 0
            while batch_idx * batch_size < X.size(0):
                lower = batch_idx * batch_size
                upper = lower + batch_size
                X_b = X[lower: upper]
                y_b = y[lower: upper]
                self.optimizer.zero_grad() # 参数梯度清零  # 6238 + 7531
                out = self.forward(X_b)  # 6238 + 19445
                # torch.cuda.empty_cache()  # 这里清也没用
                loss_b = self.loss(out, y_b)
                loss_b.backward()
                self.optimizer.step()
                loss += loss_b.item()
                pred_y = torch.max(out, dim=1).indices
                acc += torch.sum(pred_y == y_b).item()
                # print(f'Epoch {epoch}: acc: {torch.sum(pred_y == y_b).item() / batch_size}, loss: {loss_b.item()}')
                batch_idx += 1
        updates = self.get_flatten_parameters() - param
        loss /= local_epoch * (X.size(0) // batch_size)
        acc = acc / (local_epoch * X.size(0))
        if revert:
            self.load_parameters(param)
        return acc, loss, updates

    def get_flatten_parameters(self):
        """
        Return the flatten parameter of the current model
        :return: the flatten parameters as tensor
        """
        out = torch.zeros(0).to("cpu")
        with torch.no_grad():
            for parameter in self.model.parameters():
                out = torch.cat([out, parameter.flatten().to("cpu")])
        return out
    

    def get_flatten_parameters_only_length(self) -> int:
        """
        相比于get_flatten_parameters函数，这个函数不真正地将参数展开，而是只返回其总参数个数
        """
        return sum(p.numel() for p in self.model.parameters())
    

    def load_parameters(self, parameters: torch.Tensor, layers_to_ignore=[]):
        """
        Load parameters to the current model using the given flatten parameters
        :param mask: only the masked value will be loaded
        :param parameters: The flatten parameter to load
        :return: None
        """
        start_index = 0
        for name, param in self.model.named_parameters():
            with torch.no_grad():
                length = len(param.flatten())
                to_load = parameters[start_index: start_index + length].reshape(param.size())
                if not any(layer in name for layer in layers_to_ignore):
                    param.copy_(to_load.to(self.device))
            start_index += length