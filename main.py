import torch
from src import Data_Factory, FL_Torch

import argparse

# 创建解析器
parser = argparse.ArgumentParser()
# 添加 'dataset_name' 参数
parser.add_argument(
    "-d",
    "--dataset_name",
    default="mnist",
    type=str,
    help='Name of the dataset: "mnist", "cifar10", "organamnist"',
)
parser.add_argument(
    "-a",
    "--attack_mode",
    default="none",
    type=str,
    help='Mode of attack during training: "none", "scale", "grad_ascent", "mislead", "label_flip", "mix", "min_max',
)

# 解析参数
# 解析参数
args = parser.parse_args()

# 基础配置
dataset_name = args.dataset_name
Ph = 10  # 客户端数量
num_iter = 10  # 总epoch数
local_epoch = 2  # 每个客户端的local_epoch
participant_factor = 0.7  # 每轮训练的参与者所占比例
loader_batch_size = 500  # 数据加载器的batch_size（一次从loader中会获得多少数据）

# 攻击相关配置
targeted_att = args.attack_mode
malicious_factor = 0.3  # 恶意客户端的所占比例
scale_target = 0
start_attack = 0


# 防御相关的配置
layers_to_look = ["patch_embed", "attn", "head"]
kernel = 50
k_nearest = int(Ph * participant_factor) // 2 + 1
accs_mean_list = []
accs_std_list = []

# 加载数据
data_factory = Data_Factory(dataset_name, loader_batch_size, True)
trainloader, testloader = data_factory.get_loader(subset_img_num=10000)
train_img_num = data_factory.train_img_num
test_img_num = data_factory.test_img_num
trainloader_batch_num = data_factory.trainloader_batch_num
testloader_batch_num = data_factory.testloader_batch_num
classes = data_factory.classes
# trainset = torchvision.datasets.CIFAR10(
#     root="./datasets", train=True, download=True, transform=transform
# )
# testset = torchvision.datasets.CIFAR10(
#     root="./datasets", train=False, download=True, transform=transform
# )

# train_img_num = len(trainset)
# test_img_num = len(testset)

# # 创建训练子集
# subset_train_img_num = train_img_num // 5
# indices = torch.randperm(train_img_num).tolist()
# subset_indices = indices[:subset_train_img_num]
# train_subset = torch.utils.data.Subset(trainset, subset_indices)


# trainloader = torch.utils.data.DataLoader(
#     train_subset, batch_size=loader_batch_size, shuffle=True, num_workers=2
# )
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=loader_batch_size, shuffle=False, num_workers=2
# )
# trainloader_batch_num = subset_train_img_num // loader_batch_size  # 训练数据加载器的batch数量
# testloader_batch_num = test_img_num // loader_batch_size  # 测试数据加载器的batch数量

# classes = (
#     "plane",
#     "car",
#     "bird",
#     "cat",
#     "deer",
#     "dog",
#     "frog",
#     "horse",
#     "ship",
#     "truck",
# )

malicious_labels = torch.randint(0, len(classes), (train_img_num,)).to("cpu")  # 随机生成恶意标签

print(f"Data loaded, training images: {train_img_num}, testing images: {test_img_num}")


fl = FL_Torch(
    num_iter=num_iter,
    dataset_name=dataset_name,
    train_loader=trainloader,
    test_loader=testloader,
    Ph=Ph,
    malicious_labels=malicious_labels,
    malicious_factor=malicious_factor,
    participant_factor=participant_factor,
    classes=classes,
    train_img_num=train_img_num,
    test_img_num=test_img_num,
    loader_batch_size=loader_batch_size,
    trainloader_batch_num=trainloader_batch_num,
    testloader_batch_num=testloader_batch_num,
    start_attack=start_attack,
    attack_mode=targeted_att,
    k_nearest=k_nearest,
    p_kernel=kernel,
    layers_to_look=layers_to_look,
    local_epoch=local_epoch,
)

fl.print_config()
fl.federated_init()
fl.process()
# accs_mean, accs_std = fl.evaluate_all()  # 对所有模型进行测试
# print(f"ignored layers: {layers} acc_mean: {accs_mean} acc_std: {accs_std}")
# accs_mean_list.append(accs_mean)
# accs_std_list.append(accs_std)
torch.cuda.empty_cache()

# recorder = pd.DataFrame({
#     "ignored_layers": layers_to_look,
#     "acc_mean": accs_mean_list,
#     "acc_std": accs_std_list
# })
# recorder.to_csv(
#     "output/ignoredlayers_mean_std_2.csv"
# )



"""
方案1：聚类 怎么聚类？

"""