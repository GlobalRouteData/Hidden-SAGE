import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.nn.pytorch import SAGEConv
import csv
import pandas as pd
from sklearn.metrics import f1_score, recall_score


# 创建图数据


# 2014 dataset
# num_nodes = 46939
# num_nodes = 46960
# num_features = 8  # 节点特征维度  2014年没有AS霸权
#
# # message_links = '../train_data/triple_link_samples/triple_changed_message_links.txt'
# message_links = '../train_data/common_link_samples/common_changed_message_links.txt'
#
# # sample_links = '../train_data/triple_link_samples/triple_changed_sample_links.txt'
# sample_links = '../train_data/common_link_samples/common_changed_sample_links.txt'
#
# # node_features = '../train_data/triple_link_samples/triple_node_features.csv'
# node_features = '../train_data/common_link_samples/common_node_features.csv'
#
# # triple_count_file = 'triple_statics.txt'
# common_count_file = 'common_statics.txt'

# 2023 dataset
# num_nodes = 75829
num_nodes = 76187
num_features = 9  # 节点特征维度  2023年有AS霸权

# message_links = r'D:\IDEs\pycharmProjects\HiddenLinkInfer\2023\triple_changed_message_links.txt'
message_links = r'D:\IDEs\pycharmProjects\HiddenLinkInfer\2023\common_changed_message_links.txt'

# sample_links = r'D:\IDEs\pycharmProjects\HiddenLinkInfer\2023\triple_changed_sample_links.txt'
sample_links = r'D:\IDEs\pycharmProjects\HiddenLinkInfer\2023\common_changed_sample_links.txt'

# node_features = r'D:\IDEs\pycharmProjects\HiddenLinkInfer\2023\triple_node_features_1.csv'
node_features = r'D:\IDEs\pycharmProjects\HiddenLinkInfer\2023\common_node_features.csv'

# triple_count_file = '2023_triple_statics.txt'
common_count_file = '2023_common_statics.txt'

src_nodes = []
dst_nodes = []
labels = []
labeled_links = []

with open(message_links, 'r+') as f:
    for line in f:
        src_node = line.split('|')[0]
        dst_node = line.split('|')[1]
        src_nodes.append(int(src_node))
        dst_nodes.append(int(dst_node))

with open(sample_links, 'r+') as f:
    for line in f:
        line = line.replace('\n', '')
        AS1 = line.split('|')[0]
        AS2 = line.split('|')[1]
        label = line.split('|')[2]
        labeled_link = AS1 + '|' + AS2
        labeled_links.append(labeled_link)
        labels.append(int(label))

# 打乱样本，并随机取80%作为训练集，20%作为测试集
dataframe1 = pd.DataFrame()
dataframe1['labeled_links'] = labeled_links
dataframe2 = pd.DataFrame()
dataframe2['labels'] = labels
dataframe3 = pd.concat([dataframe1, dataframe2], axis=1)
train = dataframe3.sample(frac=0.8)
test = dataframe3.drop(train.index)

train_src_node = []
train_dst_node = []


# 训练集
for i in train['labeled_links']:
    AS1 = i.split('|')[0]
    AS2 = i.split('|')[1]
    train_src_node.append(int(AS1))
    train_dst_node.append(int(AS2))
train_labels = train['labels'].values.tolist()
train_src_node = torch.tensor(train_src_node)
train_dst_node = torch.tensor(train_dst_node)
train_labels = torch.tensor(train_labels)

# 测试集
test_src_node = []
test_dst_node = []
for i in test['labeled_links']:
    AS1 = i.split('|')[0]
    AS2 = i.split('|')[1]
    test_src_node.append(int(AS1))
    test_dst_node.append(int(AS2))

test_labels = test['labels'].values.tolist()
test_src_node = torch.tensor(test_src_node)
test_dst_node = torch.tensor(test_dst_node)
test_labels = torch.tensor(test_labels)


# 创建图
g = dgl.DGLGraph()
g.add_nodes(num_nodes)
g.add_edges(src_nodes, dst_nodes)

# 读取节点特征的 CSV 文件
with open(node_features, 'r') as csvfile:
    reader = csv.reader(csvfile)

    # 读取节点特征数据
    node_features_data = [list(map(float, row)) for row in reader]

# 将节点特征添加到图 g
g.ndata['feat'] = torch.tensor(node_features_data)


# 创建GraphSAGE模型
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_size, out_feats, aggregator_type='mean')

    def forward(self, g, features):
        x = torch.relu(self.conv1(g, features))
        x = self.conv2(g, x)
        return x



class EdgeClassifier(nn.Module):
    def __init__(self, in_feats, out_classes):
        super(EdgeClassifier, self).__init__()
        self.fc1 = nn.Linear(in_feats * 2, 32)
        self.fc2 = nn.Linear(32, out_classes)
        self.sig = nn.Sigmoid()

    def forward(self, src_nodes, dst_nodes, node_embed):
        # 获取节点和边的索引，然后通过索引获取相应的节点特征和边特征
        src_node_embed = node_embed[src_nodes]
        dst_node_embed = node_embed[dst_nodes]

        # 拼接两个节点的embedding和新的边特征作为边的表示
        combined_embed = torch.cat([src_node_embed, dst_node_embed], dim=-1)
        x = torch.relu(self.fc1(combined_embed))
        logits = self.fc2(x)
        return logits


# 初始化图神经网络和边分类器
graph_sage_model = GraphSAGE(num_features, 32, 16)
edge_classifier_model = EdgeClassifier(16, 1)
# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(edge_classifier_model.parameters(), lr=0.01, betas=(0.5, 0.9), weight_decay=1e-6)

node_embedding = graph_sage_model(g, g.ndata['feat'])
node_embedding = node_embedding.detach()
print("nodeebd:", node_embedding)
# 训练模型
num_epochs = 200
for epoch in range(num_epochs):

    logits = edge_classifier_model(train_src_node, train_dst_node, node_embedding)
    logits = torch.squeeze(logits)
    loss = criterion(logits, train_labels.to(torch.float32))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(edge_classifier_model.state_dict(), 'edge_classifier_model_paras.pth')

with open(common_count_file,'a+') as w:
    with torch.no_grad():
        # 获取测试集上的预测结果
        test_logits = edge_classifier_model(test_src_node, test_dst_node, node_embedding)
        test_logits = torch.sigmoid(test_logits)
        test_predicted_labels = torch.round(test_logits)
        test_predicted_labels = torch.squeeze(test_predicted_labels)
        # 计算测试集准确率
        test_accuracy = (test_predicted_labels == test_labels).float().mean().item()
        print(f'Test Accuracy: {test_accuracy:.4f}')

        # 将预测标签和真实标签转换为numpy数组
        test_predicted_labels_numpy = test_predicted_labels.cpu().numpy()
        test_labels_numpy = test_labels.cpu().numpy()

        # 计算F1 score和recall
        f1 = f1_score(test_labels_numpy, test_predicted_labels_numpy)
        recall = recall_score(test_labels_numpy, test_predicted_labels_numpy)

        # 打印结果
        print(f'F1 Score: {f1:.4f}')
        print(f'Recall: {recall:.4f}')

        w.write(str(test_accuracy)+'|'+str(f1)+'|'+str(recall)+'\n')