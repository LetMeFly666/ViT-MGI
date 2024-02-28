import torch
import torchvision
from torchvision import transforms
import medmnist
from medmnist import INFO
import numpy as np


class Data_Factory:
    def __init__(self, dataset_name, loader_batch_size, shuffle) -> None:
        self.dataset_name = dataset_name
        self.loader_batch_size = loader_batch_size
        self.shuffle = shuffle
        self.train_img_num = -1
        self.test_img_num = -1
        self.trainloader_batch_num = -1
        self.testloader_batch_num = -1
        self.transform = None
        self.classes = None

    def get_loader(self, subset_img_num=-1):
        if self.dataset_name == "mnist":
            self.transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            self.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            trainset = torchvision.datasets.MNIST(
                root="./datasets", train=True, download=True, transform=self.transform
            )
            # 加载MNIST测试数据集
            testset = torchvision.datasets.MNIST(
                root="./datasets", train=False, download=True, transform=self.transform
            )
        elif self.dataset_name == "cifar10":
            self.transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            self.classes = [
                "plane",
                "car",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ]
            trainset = torchvision.datasets.CIFAR10(
                root="./datasets", train=True, download=True, transform=self.transform
            )
            # 加载MNIST测试数据集
            testset = torchvision.datasets.CIFAR10(
                root="./datasets", train=False, download=True, transform=self.transform
            )
        elif self.dataset_name == "organamnist":
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            self.classes = [
                "bladder",
                "femur-left",
                "femur-right",
                "heart",
                "kidney-left",
                "kidney-right",
                "liver",
                "lung-left",
                "lung-right",
                "pancreas",
                "spleen",
            ]
            info = INFO[self.dataset_name]
            DataClass = getattr(medmnist, info["python_class"])
            trainset = DataClass(
                split="train",
                root="./datasets/",
                transform=self.transform,
                download=False,
                size=224,
                mmap_mode="r",
            )
            testset = DataClass(
                split="test",
                root="./datasets/",
                transform=self.transform,
                download=False,
                size=224,
                mmap_mode="r",
            )

        self.train_img_num = len(trainset)
        self.test_img_num = len(testset)
        if subset_img_num > 0 and subset_img_num <= self.train_img_num:
            indices = torch.randperm(subset_img_num).tolist()
            subset_indices = indices[:subset_img_num]
            trainset = torch.utils.data.Subset(trainset, subset_indices)
            self.train_img_num = subset_img_num

        self.trainloader_batch_num = (
            self.train_img_num + self.loader_batch_size - 1
        ) // self.loader_batch_size  # 向上取整
        self.testloader_batch_num = (
            self.test_img_num + self.loader_batch_size - 1
        ) // self.loader_batch_size  # 向上取整

        # organamnist需要特殊的处理，其标签的形状为【batch_size, 1】，需要改成【batch_size】
        if self.dataset_name == "organamnist":
            collate_fn = lambda batch: (
                torch.stack([item[0] for item in batch]),
                torch.tensor(np.array([item[1] for item in batch])).squeeze(),
            )
            trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size=self.loader_batch_size,
                shuffle=self.shuffle,
                collate_fn=collate_fn,
            )
            testloader = torch.utils.data.DataLoader(
                testset,
                batch_size=self.loader_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
        else:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=self.loader_batch_size, shuffle=self.shuffle
            )
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=self.loader_batch_size, shuffle=False
            )

        return trainloader, testloader
