from src import ModelViT
from src.Individual import Individual
import torch
import pandas as pd
import random
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
from collections import Counter

ATTACK_MODES = ["scale", "grad_ascent", "mislead", "label_flip"]


# 使用 malicious_updates + lamda * deviation
def min_max(benign_updates, model_re):
    """
    S&H attack from [4] (see Reference in readme.md), the code is authored by Virat Shejwalkar and Amir Houmansadr.
    """
    deviation = torch.std(benign_updates, 0)  # 计算良性更新沿着维度0的标准差
    lamda = torch.Tensor([10.0]).float()  # 初始值
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    distance = torch.cdist(benign_updates, benign_updates)
    max_distance = torch.max(distance)  # 计算出良性更新之间的最大距离
    del distance
    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = model_re - lamda * deviation
        distance = torch.norm((benign_updates - mal_update), dim=1) ** 2
        max_d = torch.max(distance)
        if max_d <= max_distance:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2
        lamda_fail = lamda_fail / 2
    mal_update = model_re - lamda_succ * deviation
    return mal_update


def targeted_flip(train_img: torch.Tensor, target: int, backdoor_idx=100, stride=8):
    # 将图像中的[backdoor_idx:backdoor_idx+stride, backdoor_idx:backdoor_idx+stride]这个方块的像素值设置为0.5
    augmented_data = train_img.clone()
    augmented_data[
        :, :, backdoor_idx : backdoor_idx + stride, backdoor_idx : backdoor_idx + stride
    ] = 0.5
    augmented_label = torch.ones(train_img.size(0), dtype=torch.long) * target
    return augmented_data, augmented_label


class FL_Torch:
    """
    The class handling the Federated learning process
    """

    def __init__(
        self,
        num_iter,
        dataset_name,
        train_loader,
        test_loader,
        Ph,
        classes,
        malicious_labels,
        malicious_factor,
        participant_factor,
        train_img_num,
        test_img_num,
        loader_batch_size,
        trainloader_batch_num,
        testloader_batch_num,
        start_attack,
        attack_mode,
        p_kernel,
        k_nearest,
        layers_to_look=[],
        local_epoch=4,
        output_path="./output/",
    ):
        # Number of iterations
        self.num_iter = num_iter
        # Name of dataset
        self.dataset_name = dataset_name
        # Training set features
        self.train_loader = train_loader
        # Test set features
        self.test_loader = test_loader
        # Number of participants
        self.Ph = Ph
        # Number of malicious labels
        self.malicious_labels = malicious_labels
        # Fraction of malicious participants
        self.malicious_factor = malicious_factor
        # Fraction of selected participants
        self.participant_factor = participant_factor
        # The train dataset size
        self.train_img_num = train_img_num
        # The test dataset size
        self.test_img_num = test_img_num
        # The batch size of DataLoader
        self.loader_batch_size = loader_batch_size
        # The batch number of DataLoader
        self.trainloader_batch_num = trainloader_batch_num
        self.testloader_batch_num = testloader_batch_num
        # The ignored layers
        self.layers_to_look = layers_to_look
        self.segments_of_layers = []  # layers_to_look中每一层对应参数的坐标范围
        # The batch size of paticipant
        self.batch_size = 100
        # The label classes
        self.classes = classes
        # The round when the attacker start attacking
        self.start_attack = start_attack
        # The epochs running locally on each participant at each round before sending gradients to aggregator
        self.local_epoch = local_epoch
        # Print the training information to console every 'stride' rounds
        self.stride = 2
        # The path of output files
        self.output_path = output_path
        # The k-nearest neighbours examined in the distance-based AGR (see Sec 3.4.1 of the original paper)
        self.k = k_nearest
        # The kernel size of the pooling algorithm
        self.p_kernel = p_kernel

        self.out_class = torch.tensor([len(self.classes)])
        self.global_model = ModelViT(
            "vit_base_patch16_224", False, self.out_class, "cpu"
        )
        self.participants = []
        self.loss = torch.nn.CrossEntropyLoss()
        self.collected_updates = None

        self.malicious_client = None
        self.malicious_attack = None
        self.attack_mode = attack_mode  # 攻击模式
        self.scale_target = 0

        # 主观逻辑模型
        self.slot = 4  # 主观逻辑模型更新的时隙
        self.subject_logic = None

        self.ph_batch_num = (
            self.trainloader_batch_num + self.Ph - 1
        ) // self.Ph  # 每个客户端分到的batch数量
        self.peeked_client = []  # 每一轮选择参与的客户端
        # self.is_peeked = torch.tensor(
        #     [False for i in range(self.Ph)]
        # )  # 在训练过程中，参与者是否被选上

    def federated_init(self):
        """
        Initialize FL setting, identify malicious participants
        :return: None
        """
        print("federated init......")
        param = self.global_model.get_flatten_parameters()  # 执行完就多了17G
        for i in range(self.Ph):
            model = ModelViT("vit_base_patch16_224", False, self.out_class, f"cpu")
            model.load_parameters(param)
            self.participants.append(model)
        # 恶意参与者
        sample = sorted(
            random.sample(range(self.Ph), int(self.Ph * self.malicious_factor))
        )
        self.malicious_client = [True if i in sample else False for i in range(self.Ph)]

        # 根据攻击模式为攻击者分配不同的攻击方式
        if self.attack_mode != "mix":
            self.malicious_attack = {k: self.attack_mode for k in sample}
        else:
            self.malicious_attack = {k: random.choice(ATTACK_MODES) for k in sample}

        # self.malicious_client = torch.zeros(self.Ph, dtype=torch.bool)
        # self.malicious_client.bernoulli_(self.malicious_factor)  # 伯努利采样
        print(f"malicious_participant: {self.malicious_client}")

    def print_config(self):
        print(f"num_iter: {self.num_iter}")
        print(f"dataset_name: {self.dataset_name}")

        print(f"Ph: {self.Ph}")

        print(f"malicious_factor: {self.malicious_factor}")
        print(f"participant_factor: {self.participant_factor}")

        print(f"out_class_size: {self.out_class.item()}")

        print(f"train_img_num: {self.train_img_num}")
        print(f"test_img_num: {self.test_img_num}")

        print(f"loader_batch_size: {self.loader_batch_size}")

        print(f"trainloader_batch_number: {self.trainloader_batch_num}")
        print(f"testloader_batch_number: {self.testloader_batch_num}")

        print(f"start_attack: {self.start_attack}")
        print(f"attack_mode: {self.attack_mode}")
        if self.attack_mode == "mix":
            print(f"malicious_attck: {self.malicious_attack}")

        print(f"layers_to_look: {self.layers_to_look}")
        print(f"local_epoch_of_participant: {self.local_epoch}")
        print(f"attack_mode: {self.attack_mode}")
        if self.attack_mode == "scale":
            print(f"scale_target: {self.scale_target}")

        print(f"ph_batch_num: {self.ph_batch_num}")
        print(f"batch_size_of_participant: {self.batch_size}")

    def reset_collected_updates(self):
        """
        Reset the globally collected gradients
        :return:
        """
        print("reset global updates......")
        if self.collected_updates is None:
            length = self.global_model.get_flatten_parameters().size(0)
            self.collected_updates = torch.zeros(self.Ph, length).to("cpu")
        else:
            self.collected_updates.zero_()

    def collect_updates(self, idx: int, local_update: torch.tensor):
        """
        AGR collect gradients from the participants
        :param idx: The index of the participant
        :param local_grad: the local gradients from the participant
        :return: None
        """
        self.collected_updates[idx] = local_update

    def back_prop(self, attack=False):
        """
        Conduct back propagation of one specific participant
        :param attack: if the attacker starts attacking
        :param attack_mode: the type of attack conducted by the attacker
        """
        sum_acc = 0
        sum_loss = 0
        peeked_list = [
            True if i in self.peeked_client else False for i in range(self.Ph)
        ]
        for i, data in enumerate(self.train_loader):
            # 根据i计算出相应客户端
            client_id = i // self.ph_batch_num  # 计算出data_i属于的参与者
            if not peeked_list[client_id]:
                continue
            model = self.participants[client_id]
            X, y = data
            # print(X.shape, y.shape)
            acc, loss, updates = model.train(
                X.to(f"cpu"), y.to(f"cpu"), self.batch_size, self.local_epoch
            )
            print(
                f"normal participant {client_id} train on data {i}: acc: {acc}, loss: {loss}"
            )
            updates = updates.to("cpu")
            self.collect_updates(client_id, updates)
            sum_acc += acc
            sum_loss += loss

            # 参与者展开攻击
            if (
                attack
                and self.malicious_client[client_id]
                and self.malicious_attack[client_id] in ["grad_ascent"]
            ):
                print(
                    f"malicious participant {client_id} launch {self.malicious_attack[client_id]} attack"
                )
                local = self.collected_updates[client_id]  # 客户端对应的正常更新
                mal_update = -local
                self.collect_updates(client_id, mal_update)

            if (
                attack
                and self.malicious_client[client_id]
                and self.malicious_attack[client_id] in ["mislead", "label_flip"]
            ):
                y = self.malicious_labels[
                    i * (X.size(0)) : (i + 1) * (X.size(0))
                ]  # 置换标签
                acc, loss, updates = model.train(
                    X.to(f"cpu"), y.to(f"cpu"), self.batch_size, self.local_epoch
                )
                print(
                    f"malicious participant {client_id} launch {self.malicious_attack[client_id]} attack"
                )
                updates = updates.to("cpu")
                if self.malicious_attack[client_id] == "label_flip":
                    mal_update = updates
                else:
                    local = self.collected_updates[client_id]
                    mal_update = updates - local
                self.collect_updates(client_id, mal_update)

            if (
                attack
                and self.malicious_client[client_id]
                and self.malicious_attack[client_id] in ["scale"]
            ):
                # Conduct T-scal attack from [1]
                X, y = targeted_flip(X, self.scale_target)
                acc, loss, updates = model.train(
                    X.to(f"cpu"), y.to(f"cpu"), self.batch_size, self.local_epoch
                )
                print(
                    f"malicious participant {client_id} launch {self.malicious_attack[client_id]} attack"
                )
                updates = updates.to("cpu")
                local = self.collected_updates[client_id]
                mal_updates = local + updates / self.malicious_factor
                self.collect_updates(client_id, mal_updates)

        if attack and self.attack_mode == "min_max":
            # Call the code snip from [4] to conduct S&H attack
            all_updates = self.collected_updates.clone()
            # malicious_client_tensor = torch.tensor(self.malicious_client, dtype=torch.bool)
            # 获得所有良性节点的下标
            benign_client_indices = [
                i
                for i, is_malicious in enumerate(self.malicious_client)
                if not is_malicious
            ]
            benign_updates = all_updates[benign_client_indices]  # 获得良性节点的更新
            for i in range(self.Ph):
                if not self.malicious_client[i]:
                    continue
                local = self.collected_updates[i]
                mal_grad = min_max(benign_updates, local)
                self.collect_updates(i, mal_grad)
        return (sum_acc / int(self.Ph * self.participant_factor) / self.ph_batch_num), (
            sum_loss / int(self.Ph * self.participant_factor) / self.ph_batch_num
        )

    def send_param(self):
        """
        Participants collect the parameters from the global model
        :param sparsify: Not used, if apply sparsify update
        :return: None
        """
        param = self.global_model.get_flatten_parameters()
        for i in range(self.Ph):
            self.participants[i].load_parameters(param)

    def apply_updates(self, updates: torch.tensor):
        """
        Apply the collected gradients to the global model
        :return: None
        """
        print("apply updates......")
        model = self.global_model
        avg_update = torch.mean(updates)
        param = model.get_flatten_parameters()
        param = param + avg_update
        model.load_parameters(param)

    def extract_param_by_layers(
        self,
    ):
        # 对客户端上传的更新提取出相应层的参数，拼接为一个向量
        peeked_updates = self.collected_updates[self.peeked_client]
        if len(self.segments_of_layers) == 0:
            start_index = 0
            for name, param in self.global_model.model.named_parameters():
                # print(name, param.shape)a
                with torch.no_grad():
                    length = len(param.flatten())
                    if any(layer in name for layer in self.layers_to_look):
                        self.segments_of_layers.append(
                            (start_index, start_index + length)
                        )
                start_index += length
        extract_params = []
        for update in peeked_updates:
            segment_list = [update[start:end] for start, end in self.segments_of_layers]
            extract_params.append(torch.cat(segment_list))

        return torch.stack(
            extract_params
        )  # torch.stack() 创建的是一个新的张量，并不与 self.collected_updates 中的数据共享内存

    # 对参数进行池化操作(一维池化)
    def normal_pooling(self, updates: torch.Tensor, kernel_size=3):
        updates = updates.unsqueeze(1)
        stride = kernel_size
        pooled = F.max_pool1d(updates, kernel_size=kernel_size, stride=stride)
        return pooled.squeeze(1)

    # 计算cosin相似度
    def cosine_distance_torch(self, updates: torch.Tensor, eps=1e-8):
        w = updates.norm(p=2, dim=1, keepdim=True)  # norm of each row of x1
        return torch.mm(updates, updates.T) / (w * w.T).clamp(
            min=eps
        )  # cosine similarity

    # 计算k_nearest
    def malicious_filter(self, cosin_matrix: torch.Tensor):
        k_nearest = torch.topk(cosin_matrix, k=self.k, dim=1)
        neighbour_dist = torch.zeros(cosin_matrix.size(0))
        for i in range(cosin_matrix.size(0)):
            idx = k_nearest.indices[i]
            neighbour = cosin_matrix[idx][:, idx]
            neighbour_dist[i] = neighbour.sum()
        return neighbour_dist

    # def evaluate_all(self):
    #     # 利用测试数据对所有模型进行评估，计算出所有模型准确性的均值和标准差
    #     # 只评价参与训练过的模型
    #     cnt = self.is_peeked.sum().item()
    #     clients = torch.where(self.is_peeked)[0]
    #     accs = torch.zeros(cnt + 1)
    #     for index, client in enumerate(clients):
    #         for data in self.test_loader:
    #             test_x, test_y = data  # 测试数据
    #             test_x = test_x.to(f"cpu")
    #             test_y = test_y.to(f"cpu")
    #             model = self.participants[client]  # 第i个参与者训练出的模型
    #             with torch.no_grad():
    #                 out = model.forward(test_x)  # 预测输出
    #                 pred_y = torch.max(out, dim=1).indices
    #                 accs[index] += torch.sum(pred_y == test_y).item()
    #         accs[index] /= self.test_img_num
    #     accs[cnt], _ = self.evaluate_global()
    #     return torch.mean(accs), torch.std(accs)

    def evaluate_global(self):
        """
        Evaluate the global model accuracy and loss value
        :return: accuracy and loss value
        """
        total_loss = 0
        total_acc = 0
        for data in self.test_loader:
            test_x, test_y = data  # 测试数据
            test_x = test_x.to("cpu")
            test_y = test_y.to("cpu")
            model = self.global_model
            with torch.no_grad():
                out = model.forward(test_x)  # 预测输出
                total_loss += self.loss(out, test_y).item()
                pred_y = torch.max(out, dim=1).indices
                total_acc += torch.sum(pred_y == test_y).item()
        return total_acc / self.test_img_num, total_loss / self.testloader_batch_num

    def evaluate_target(self):
        """
        Evaluate loss value and accuracy of the targeted label
        :return: accuracy and loss value
        """
        total_loss = 0
        total_acc = 0
        total_img_num = 0  # 测试集除去标签等于scale_target后的总数
        model = self.global_model
        for test_x, test_y in self.test_loader:
            test_x = test_x.to("cpu")
            test_y = test_y.to("cpu")
            mal_test_x, mal_test_y = targeted_flip(
                test_x, self.scale_target
            )  # 得到标记后的图像以及相应标签
            with torch.no_grad():
                out = model.forward(mal_test_x)  # 预测输出
                total_loss += self.loss(out, mal_test_y).item()
                pred_y = torch.max(out, dim=1).indices
                idx = test_y != self.scale_target  # 筛选出标签不等于scale_target的测试数据
                pred_y = pred_y[idx]
                test_y = test_y[idx]
                total_img_num += test_y.size(0)
                total_acc += torch.sum(pred_y == self.scale_target).item()
        return total_acc / total_img_num, total_loss / self.testloader_batch_num

    def process(self):
        """
        Organize the FL training process
        """
        epoch_col = []
        train_acc_col = []
        train_loss_col = []
        test_acc_col = []
        test_loss_col = []
        attacking = False
        clients_status = []
        participant_num = int(self.Ph * self.participant_factor)
        for epoch in range(self.num_iter):
            print(f"Main epoch{epoch + 1}: start to send params")
            self.send_param()  # 将全局模型的参数应用到每个参与者
            self.reset_collected_updates()
            if epoch % self.slot == 0:
                # 更新主观逻辑模型
                if epoch == 0:
                    self.subject_logic = [Individual(i) for i in range(self.Ph)]
                    #  初始，随机选择本轮的参与者
                    self.peeked_client = sorted(
                        random.sample(range(self.Ph), participant_num)
                    )
                else:
                    assert len(clients_status) == self.slot
                    # 更新主观逻辑模型，选择信誉高的参与者
                    matrix = np.array(clients_status)
                    diff = np.diff(matrix, axis=0)  # 计算每一列的相邻元素之间的差异
                    jump_counts = np.sum(np.abs(diff) == 1, axis=0)  # 计算每列的跳变次数
                    ones_count = np.sum(matrix == 1, axis=0)  # 统计每列中1出现的次数
                    zeros_count = np.sum(matrix == 0, axis=0)  # 统计每列中0出现的次数
                    print(f"jump counts: {jump_counts}")
                    print(f"ones counts: {ones_count}")
                    for i in range(participant_num):  # 更新参与者的信誉，没有参与的参与者继承之前的信誉值
                        new_b = (1 - 0.1) * (
                            0.4
                            * ones_count[i]
                            / (0.4 * ones_count[i] + 0.6 * zeros_count[i])
                        )  # belive
                        new_d = (1 - 0.1) * (
                            0.6
                            * zeros_count[i]
                            / (0.4 * ones_count[i] + 0.6 * zeros_count[i])
                        )  # disbelive
                        assert (
                            self.subject_logic[self.peeked_client[i]].id
                            == self.peeked_client[i]
                        )
                        self.subject_logic[self.peeked_client[i]].update_param(
                            new_b, new_d, 0.1
                        )
                    sorted_clients = sorted(
                        self.subject_logic[:],
                        key=lambda individual: individual.get_reputation_value(),
                        reverse=True,
                    )
                    self.peeked_client = sorted(
                        [
                            individual.id
                            for individual in sorted_clients[:participant_num]
                        ]
                    )  # 选择信誉高的参与者
                print("subjective logic:")
                for individual in self.subject_logic:
                    print(individual)
                clients_status = []  # 清空记录

            # self.is_peeked[self.peeked_client] = True
            print(f"peeked participants: {self.peeked_client}")
            if epoch == self.start_attack:
                attacking = True
                print(f"Start attacking at epoch {epoch}")
            acc, loss = self.back_prop(attacking)

            # 1. 计算不提取相应层时的准确度
            print("Not extract but Pooled")
            peeked_updates = self.collected_updates[self.peeked_client].clone()
            pooled_updates = self.normal_pooling(peeked_updates, self.p_kernel)
            cosin_matirx = self.cosine_distance_torch(pooled_updates)
            print(f"cosin_matirx:\n{cosin_matirx}")
            similarity_score = self.malicious_filter(cosin_matirx)  # 根据matirx计算相似分数
            print(f"similarity_score: {similarity_score}")
            normalized_similarity_score = torch.nn.functional.softmax(
                similarity_score, dim=0
            )  # 利用softmax进行标准化
            print(f"normalized_similarity_score: {normalized_similarity_score}")
            # 基于最大距离的层次聚类算法
            Z = linkage(
                normalized_similarity_score.numpy().reshape(-1, 1), method="average"
            )
            clusters = fcluster(Z, 0.01, criterion="distance")
            print(clusters)

            # 2. 计算不提取相应层且不池化的准确度
            print("Not extract and Not Pooled")
            peeked_updates = self.collected_updates[self.peeked_client].clone()
            cosin_matirx = self.cosine_distance_torch(peeked_updates)
            print(f"cosin_matirx:\n{cosin_matirx}")
            similarity_score = self.malicious_filter(cosin_matirx)  # 根据matirx计算相似分数
            print(f"similarity_score: {similarity_score}")
            normalized_similarity_score = torch.nn.functional.softmax(
                similarity_score, dim=0
            )  # 利用softmax进行标准化
            print(f"normalized_similarity_score: {normalized_similarity_score}")
            # 基于最大距离的层次聚类算法
            Z = linkage(
                normalized_similarity_score.numpy().reshape(-1, 1), method="complete"
            )
            clusters = fcluster(Z, 0.01, criterion="distance")
            print(clusters)

            # 3. 计算提取相应层但不池化的准确度
            print("Extracted but Not Pooled")
            extracted_updates = self.extract_param_by_layers()
            cosin_matirx = self.cosine_distance_torch(extracted_updates)
            print(f"cosin_matirx:\n{cosin_matirx}")
            similarity_score = self.malicious_filter(cosin_matirx)  # 根据matirx计算相似分数
            print(f"similarity_score: {similarity_score}")
            normalized_similarity_score = torch.nn.functional.softmax(
                similarity_score, dim=0
            )  # 利用softmax进行标准化
            print(f"normalized_similarity_score: {normalized_similarity_score}")
            # 基于最大距离的层次聚类算法
            Z = linkage(
                normalized_similarity_score.numpy().reshape(-1, 1), method="complete"
            )
            clusters = fcluster(Z, 0.01, criterion="distance")
            print(clusters)

            # 4. 计算提取相应层且池化的准确度
            print("Extracted and Pooled")
            extracted_updates = self.extract_param_by_layers()
            pooled_updates = self.normal_pooling(extracted_updates, self.p_kernel)
            cosin_matirx = self.cosine_distance_torch(pooled_updates)
            print(f"cosin_matirx:\n{cosin_matirx}")
            similarity_score = self.malicious_filter(cosin_matirx)  # 根据matirx计算相似分数
            print(f"similarity_score: {similarity_score}")
            normalized_similarity_score = torch.nn.functional.softmax(
                similarity_score, dim=0
            )  # 利用softmax进行标准化
            print(f"normalized_similarity_score: {normalized_similarity_score}")
            # 基于最大距离的层次聚类算法
            Z = linkage(
                normalized_similarity_score.numpy().reshape(-1, 1), method="complete"
            )
            clusters = fcluster(Z, 0.01, criterion="distance")
            print(clusters)

            cluster_counts = Counter(clusters)  # 统计每个聚类的数量
            max_cluster = cluster_counts.most_common(1)[0][0]  # 找到数量最多的聚类
            max_cluster_indices = np.where(clusters == max_cluster)[
                0
            ]  # 获取属于最大聚类的数据点的下标
            final_updates = self.collected_updates[max_cluster_indices].clone()
            self.apply_updates(final_updates)  # 聚合更新并应用到全局模型中

            # 记录当次状态
            status = [
                1 if i in max_cluster_indices else 0
                for i in range(int(self.Ph * self.participant_factor))
            ]
            print(f"benign participant: {status}")

            # 更新主观逻辑模型
            clients_status.append(status)
            # Print the training progress every 'stride' rounds
            if (epoch + 1) % self.stride == 0:
                if attacking and self.attack_mode == "scale":
                    test_acc, test_loss = self.evaluate_target()
                    print(
                        f"Epoch {epoch+1} - attack acc {test_acc:6.4f}, test loss: {test_loss:6.4f}, train acc {acc:6.4f}"
                        f", train loss {loss:6.4f}"
                    )
                else:
                    test_acc, test_loss = self.evaluate_global()
                    print(
                        f"Epoch {epoch+1} - test acc {test_acc:6.4f}, test loss: {test_loss:6.4f}, train acc {acc:6.4f}"
                        f", train loss {loss:6.4f}"
                    )
                epoch_col.append(epoch)
                test_acc_col.append(test_acc)
                test_loss_col.append(test_loss)
                train_acc_col.append(acc)
                train_loss_col.append(loss)
        recorder = pd.DataFrame(
            {
                "epoch": epoch_col,
                "test_acc": test_acc_col,
                "test_loss": test_loss_col,
                "train_acc": train_acc_col,
                "train_loss": train_loss_col,
            }
        )
        recorder.to_csv(
            self.output_path
            + f"{self.dataset_name}_Ph_{self.Ph}_MF_{self.malicious_factor}_K_{self.p_kernel}"
            + f"_attack_{self.attack_mode}_start_{self.start_attack}"
            + ".csv"
        )
