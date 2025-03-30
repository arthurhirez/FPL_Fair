import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch
from utils.finch import FINCH
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedHierarchy.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos


class FPL(FederatedModel):
    NAME = 'fpl'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FPL, self).__init__(nets_list, args, transform)
        self.global_protos = [] # lista #classes prototipos globais
        self.local_protos = {} # dict #clientes : lista #classes prototipos locais

        self.global_history = [] # comm_epoch : deepcopy(global_protos)
        self.local_history = {idx: [] for idx in range(self.online_num)} # comm_epoch : {client : {local_epoch : deepcopy(local_protos)}}
        self.local_metrics_train = {idx: [] for idx in range(self.online_num)}
        self.local_metrics_test = {idx: [] for idx in range(self.online_num)}

        self.infoNCET = args.infoNCET

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def proto_aggregation(self, local_protos_list):
        # local_protos_list = self.local_protos
        # original_data = copy.deepcopy(avaliable_indexes[domain])
        agg_protos_label = dict()

        # Iterate through online clients, grouping local protos by label (all domains)
        for idx in self.online_clients:
            local_protos = local_protos_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        # Perform clustering and aggregation for each label
        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                # Convert prototype tensors to numpy arrays for clustering
                proto_list = [item.squeeze(0).detach().cpu().numpy().reshape(-1) for item in proto_list]
                proto_list = np.array(proto_list)

                # Apply FINCH clustering algorithm on the prototype list
                c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
                                            ensure_early_exit=False, verbose=True)

                m, n = c.shape
                class_cluster_list = []

                # Extract the last column of the clustering matrix, which represents final cluster assignments
                for index in range(m):
                    class_cluster_list.append(c[index, -1])

                class_cluster_array = np.array(class_cluster_list)
                uniqure_cluster = np.unique(class_cluster_array).tolist()
                agg_selected_proto = []  # List to store aggregated prototypes for each cluster

                for _, cluster_index in enumerate(uniqure_cluster):
                    # Get indices of prototypes belonging to the current cluster
                    selected_array = np.where(class_cluster_array == cluster_index)

                    # Extract prototypes belonging to the cluster
                    selected_proto_list = proto_list[selected_array]

                    # Compute mean prototype for the cluster
                    proto = np.mean(selected_proto_list, axis=0, keepdims=True)

                    # Convert to a tensor and store
                    agg_selected_proto.append(torch.tensor(proto))

                # Store the aggregated prototypes for this label
                agg_protos_label[label] = agg_selected_proto
            else:
                agg_protos_label[label] = [proto_list[0].data]

        # self.global_history.append(agg_protos_label.detach().cpu().numpy())
        self.global_history.append({
                        key: [t.detach().cpu().numpy() for t in tensor_list]  # Convert each tensor
                        if isinstance(tensor_list, list) else tensor_list.detach().cpu().numpy()  # Handle single tensors
                        for key, tensor_list in agg_protos_label.items()
                    })

        return agg_protos_label

    def hierarchical_info_loss(self, f_now, label, all_f, mean_f, all_global_protos_keys):
        # print("\n=== DEBUGGING INFO ===")
        #
        # # Print type and shape of all_f elements
        # print("Type of all_f:", type(all_f))
        # if isinstance(all_f, list):
        #     print(f"all_f contains {len(all_f)} elements.")
        #     for i, item in enumerate(all_f):
        #         print(f"Element {i}: Type={type(item)}, Shape={item.shape if hasattr(item, 'shape') else 'N/A'}")
        # elif isinstance(all_f, torch.Tensor):
        #     print(f"all_f is a tensor with shape: {all_f.shape}")
        # elif isinstance(all_f, np.ndarray):
        #     print(f"all_f is a numpy array with shape: {all_f.shape}")
        #
        # # Print type and shape of all_global_protos_keys
        # print("\nType of all_global_protos_keys:", type(all_global_protos_keys))
        # if isinstance(all_global_protos_keys, torch.Tensor):
        #     print(f"Shape of all_global_protos_keys torch.Tensor: {all_global_protos_keys.shape}")
        # elif isinstance(all_global_protos_keys, np.ndarray):
        #     print(f"Shape of all_global_protos_keys np.array: {all_global_protos_keys.shape}")
        #
        # # Print label info
        # print("\nType of label:", type(label))
        # if isinstance(label, torch.Tensor):
        #     print(f"Label torch.Tensor value: {label} {label.item()}, Shape: {label.shape}")
        # elif isinstance(label, np.ndarray):
        #     print(f"Label np.ndarray value: {label}, Shape: {label.shape}")
        #
        #
        # print("\nType of meanf:", type(mean_f))
        #
        # print(f"meanf torch.Tensor value: Shape: {len(mean_f)}")

        #
        # # Check boolean indexing
        # try:
        #     indices = (all_global_protos_keys == label.item()).nonzero()
        #     print(f"\nNumber of matching indices: {len(indices)}")
        #     if len(indices) > 0:
        #         print(f"First matching index: {indices[0]}")
        #         print(f"Indexes: {indices}")
        # except Exception as e:
        #     print("\nError while checking indices:", e)
        #
        # print("\n======================\n")

        # f_pos = np.array(all_f)[all_global_protos_keys == label.item()][0].to(self.device)
        # f_neg = torch.cat(list(np.array(all_f)[all_global_protos_keys != label.item()])).to(self.device)

        # f_pos = [f for i, f in enumerate(all_f) if all_global_protos_keys[i] == label.item()][0].to(self.device)
        # f_neg = torch.cat([f for i, f in enumerate(all_f) if all_global_protos_keys[i] != label.item()]).to(self.device)
        # xi_info_loss = self.calculate_infonce(f_now, f_pos, f_neg)
        #
        #
        # # mean_f_pos = np.array(mean_f)[all_global_protos_keys == label.item()][0].to(self.device)
        # mean_f_pos = [f for i, f in enumerate(all_f) if all_global_protos_keys[i] == label.item()][0].to(self.device)
        # mean_f_pos = mean_f_pos.view(1, -1)

        f_idx = np.where(all_global_protos_keys == label.item())[0][0]
        f_pos = all_f[f_idx].to(self.device)
        f_neg = torch.cat([f for i, f in enumerate(all_f) if i != f_idx]).to(self.device)
        xi_info_loss = self.calculate_infonce(f_now, f_pos, f_neg)

        mean_f_pos = mean_f[f_idx].to(self.device)

        # mean_f_neg = torch.cat(list(np.array(mean_f)[all_global_protos_keys != label.item()]), dim=0).to(self.device)
        # mean_f_neg = mean_f_neg.view(9, -1)

        loss_mse = nn.MSELoss()
        cu_info_loss = loss_mse(f_now, mean_f_pos)

        hierar_info_loss = xi_info_loss + cu_info_loss
        return hierar_info_loss

    def calculate_infonce(self, f_now, f_pos, f_neg):
        f_proto = torch.cat((f_pos, f_neg), dim=0)
        l = torch.cosine_similarity(f_now, f_proto, dim=1)
        l = l / self.infoNCET

        exp_l = torch.exp(l)
        exp_l = exp_l.view(1, -1)
        pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(self.device)
        pos_mask = pos_mask.view(1, -1)
        # pos_l = torch.einsum('nc,ck->nk', [exp_l, pos_mask])
        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)
        infonce_loss = -torch.log(sum_pos_l / sum_exp_l)
        return infonce_loss

    def loc_update(self, priloader_list, testloader_list, mapped_indexes, epoch_index):
        total_clients = list(range(self.args.parti_num))
        # online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        online_clients = total_clients
        self.online_clients = total_clients

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i][epoch_index])
        self.global_protos = self.proto_aggregation(self.local_protos)
        self.aggregate_nets(freq=None, epoch_index=epoch_index)

        # Evaluate each client on test data
        for i in online_clients:
            self.evaluate_client(i, self.nets_list[i], priloader_list[i][epoch_index], train_test = 'train')
            self.evaluate_client(i, self.nets_list[i], testloader_list[mapped_indexes[i]], train_test = 'test')

        return None


    
    def evaluate_client(self, index, net, data_loader, train_test):
        """
        Evaluate a trained client model on its test dataset and compute multiple metrics per label.
    
        Args:
            index (int): Client index.
            net (torch.nn.Module): The trained client network.
            data_loader (DataLoader): The test dataset.
            train_test (str): Indicates whether it's train or test evaluation.
    
        Returns:
            dict: Dictionary containing overall accuracy and per-label metrics.
        """
        net.eval()  # Set model to evaluation mode
    
        all_labels = []
        all_preds = []
    
        with torch.no_grad():  # Disable gradient computation for efficiency
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)  # Get predicted class
    
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
    
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        # Compute overall accuracy
        acc = accuracy_score(all_labels, all_preds) * 100
    
        # Compute per-label accuracy
        unique_labels = np.unique(all_labels)  # Get unique class labels
        per_label_accuracy = {}
        
        for label in unique_labels:
            mask = all_labels == label
            correct = np.sum(all_preds[mask] == label)
            total = np.sum(mask)
            per_label_accuracy[label] = (correct / total) * 100 if total > 0 else 0
    
        # Compute per-label precision, recall, and F1-score
        precision_per_label = precision_score(all_labels, all_preds, average=None, zero_division=0) * 100
        recall_per_label = recall_score(all_labels, all_preds, average=None, zero_division=0) * 100
        f1_per_label = f1_score(all_labels, all_preds, average=None, zero_division=0) * 100
    
        # Compute macro-averaged metrics
        precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
        recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    
        # Store metrics
        client_metrics = {
            "accuracy": acc,
            "macro_precision": precision_macro,
            "macro_recall": recall_macro,
            "macro_f1-score": f1_macro,
            "per_label_metrics": {
                "accuracy": per_label_accuracy,
                "precision": precision_per_label.tolist(),
                "recall": recall_per_label.tolist(),
                "f1-score": f1_per_label.tolist()
            }
        }
    
        if train_test.lower() == 'train':
            self.local_metrics_train[index].append(client_metrics)
        elif train_test.lower() == 'test':
            self.local_metrics_test[index].append(client_metrics)
        else:
            raise ValueError("Invalid train_test parameter. Use 'train' or 'test'.")
    
        net.train()  # Restore model to training mode
        return client_metrics


    # def evaluate_client(self, index, net, data_loader, train_test):
    #     """
    #     Evaluate a trained client model on its test dataset and compute multiple metrics.

    #     Args:
    #         index (int): Client index.
    #         net (torch.nn.Module): The trained client network.
    #         data_loader (DataLoader): The test dataset.

    #     Returns:
    #         dict: Dictionary containing accuracy, precision, recall, and F1-score.
    #     """
    #     net.eval()  # Set model to evaluation mode

    #     all_labels = []
    #     all_preds = []

    #     with torch.no_grad():  # Disable gradient computation for efficiency
    #         for images, labels in data_loader:
    #             images, labels = images.to(self.device), labels.to(self.device)
    #             outputs = net(images)
    #             _, predicted = torch.max(outputs, 1)  # Get predicted class

    #             all_labels.extend(labels.cpu().numpy())
    #             all_preds.extend(predicted.cpu().numpy())

    #     # Compute evaluation metrics
    #     acc = accuracy_score(all_labels, all_preds) * 100
    #     precision = precision_score(all_labels, all_preds, average = "macro", zero_division = 0) * 100
    #     recall = recall_score(all_labels, all_preds, average = "macro", zero_division = 0) * 100
    #     f1 = f1_score(all_labels, all_preds, average = "macro", zero_division = 0) * 100

    #     # Store metrics
    #     client_metrics = {"accuracy": acc, "precision": precision, "recall": recall, "f1-score": f1}

    #     # print(f"Client {index} | Acc: {acc:.2f}% | Prec: {precision:.2f}% | Rec: {recall:.2f}% | F1: {f1:.2f}%")

    #     if train_test.lower() == 'train':
    #         self.local_metrics_train[index].append(client_metrics)
    #     elif train_test.lower() == 'test':
    #         self.local_metrics_test[index].append(client_metrics)
    #     else:
    #         raise ValueError("Invalid train_test parameter. Use 'train' or 'test'.")

    #     net.train()  # Restore model to training mode
    #     return client_metrics

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)

        if len(self.global_protos) != 0:
            all_global_protos_keys = np.array(list(self.global_protos.keys()))
            all_f = []
            mean_f = []
            for protos_key in all_global_protos_keys:
                temp_f = self.global_protos[protos_key]
                temp_f = torch.cat(temp_f, dim=0).to(self.device)
                all_f.append(temp_f.cpu())
                mean_f.append(torch.mean(temp_f, dim=0).cpu())
            all_f = [item.detach() for item in all_f]
            mean_f = [item.detach() for item in mean_f]

        iterator = tqdm(range(self.local_epoch))
        for iter in iterator:
            agg_protos_label = {}
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                images = images.to(self.device)
                labels = labels.to(self.device)
                f = net.features(images)
                outputs = net.classifier(f)

                lossCE = criterion(outputs, labels)

                if len(self.global_protos) == 0:
                    loss_InfoNCE = 0 * lossCE
                else:
                    i = 0
                    loss_InfoNCE = None

                    for label in labels:
                        if label.item() in self.global_protos.keys():
                            f_now = f[i].unsqueeze(0)
                            loss_instance = self.hierarchical_info_loss(f_now, label, all_f, mean_f, all_global_protos_keys)

                            if loss_InfoNCE is None:
                                loss_InfoNCE = loss_instance
                            else:
                                loss_InfoNCE += loss_instance
                        i += 1
                    loss_InfoNCE = loss_InfoNCE / i
                loss_InfoNCE = loss_InfoNCE

                loss = lossCE + loss_InfoNCE
                loss.backward()
                iterator.desc = "Local Pariticipant %d CE = %0.3f,InfoNCE = %0.3f" % (index, lossCE, loss_InfoNCE)
                optimizer.step()

                if iter == self.local_epoch - 1:
                    for i in range(len(labels)):
                        if labels[i].item() in agg_protos_label:
                            agg_protos_label[labels[i].item()].append(f[i, :])
                        else:
                            agg_protos_label[labels[i].item()] = [f[i, :]]

        agg_protos = agg_func(agg_protos_label)

        # self.local_history[index].append(copy.deepcopy(agg_protos))
        self.local_history[index].append({
                                        key: tensor.detach().cpu().numpy()
                                        for key, tensor in agg_protos.items()
                                    })

        self.local_protos[index] = agg_protos