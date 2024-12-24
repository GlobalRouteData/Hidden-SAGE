import time
import networkx as nx
from collections import defaultdict
import os
import json
import csv
import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.nn.pytorch import SAGEConv
import csv
import pandas as pd
from sklearn.metrics import f1_score, recall_score
import pickle

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

class Infer_Hidden_Links():

    def __init__(self,source_data_folder,progress_data_folder,feature_data_folder,result_data_folder):
        # input
        self.source_data_folder = source_data_folder
        self.progress_data_folder = progress_data_folder
        self.feature_data_folder = feature_data_folder
        self.result_data_folder = result_data_folder

        # output
        self.changed_all_links_file = os.path.join(self.progress_data_folder,'changed_links_file.txt')
        self.sorted_node_file = os.path.join(self.progress_data_folder,'sorted_node_file.txt')

    def AS2num(self):
        '''
        将AS映射为顺序数字
        :return:
        '''
        print("开始寻找BGP原始文件...")
        for file_name in os.listdir(self.source_data_folder):
            if 'all-paths.txt' in file_name: # 寻找BGP原始文件
                all_links_file = os.path.join(self.source_data_folder,file_name)
        print(f"成功！原始文件名：{all_links_file}")
        ASes = set()
        node_index = defaultdict(int)
        print("开始排序...")
        with open(all_links_file, 'r+') as f:
            for line in f:
                line = line.replace('\n', '')
                try:
                    AS1 = int(line.split('|')[0])
                    AS2 = int(line.split('|')[1])
                    ASes.add(AS1)
                    ASes.add(AS2)
                except:
                    continue
        ASes = sorted(list(ASes))
        print("排序成功！开始映射...")
        with open(self.sorted_node_file, 'w+') as w:
            for i in range(len(ASes)):
                print(ASes[i], i)
                node_index[ASes[i]] = i
                # 映射文件  格式为 <原ASN>|<映射后序号>
                w.write(str(ASes[i]) + '|'+str(i)+'\n')
        with open(self.changed_all_links_file, 'w+') as w:
            with open(all_links_file, 'r+') as f:
                for line in f:
                    try:
                        AS1 = int(line.split('|')[0])
                        AS2 = int(line.split('|')[1])
                    except:
                        continue
                    AS1_index = str(node_index[AS1])
                    AS2_index = str(node_index[AS2])
                    new_link = AS1_index + '|' + AS2_index
                    w.write(new_link + '\n')
        print("映射成功！")

    def create_new_links(self):  # 大约2小时完成
        '''
        将所有节点排列组合，得到可见链接之外的所有边

        create_new_links.exe -fileA <文件A路径> -fileB <文件B路径>
        :return:
        '''
        t1 = time.time()
        input_file_path = self.changed_all_links_file
        output_file_path = os.path.join(self.progress_data_folder,'new_links.txt')
        command = f'create_new_links.exe -fileA={input_file_path} -fileB={output_file_path}'
        os.system(command)
        t2 = time.time()
        t3 = t2-t1
        with open('create_new_links_time.txt','w+') as f:
            f.write("共耗时："+str(t3))

    def create_features(self):
        '''
        构建所有AS节点的特征向量

        :return: 按ASN从小到大顺序排好的节点特征csv
        '''
        # input
        degree_file_path = os.path.join(self.feature_data_folder, 'degree.json')
        transit_file_path = os.path.join(self.feature_data_folder, 'transit_degree.json')
        AS_hierarchy_file_path = os.path.join(self.feature_data_folder, 'AS_hierarchy.json')
        AS_pagerank_file_path = os.path.join(self.feature_data_folder, 'AS_pagerank.json')
        AS_Type_file_path = os.path.join(self.feature_data_folder, 'AS_Type.json')
        customer_cone_file_path = os.path.join(self.feature_data_folder, 'customer_cone.json')
        AS_geolocation_file_path = os.path.join(self.feature_data_folder, 'AS_geolocation_code.json')
        high_hierarchy_num_file_path = os.path.join(self.feature_data_folder, 'high_hierarchy_neighbour.json')
        AS_hegemony_file_path = os.path.join(self.feature_data_folder, 'AS_hegemony.json')

        # output
        output_file_path = os.path.join(self.feature_data_folder,'node_features.csv')

        print('开始构建特征集...')
        for file_name in os.listdir(self.progress_data_folder):
            if 'sorted_node' in file_name: # 寻找排序后的节点文件
                sorted_node_file = os.path.join(self.progress_data_folder,file_name)
        ASN_list = []
        with open(sorted_node_file, 'r+') as f:
            for line in f:
                ASN = line.replace('\n', '').split('|')[0]
                ASN_list.append(ASN)
        with open(degree_file_path, 'r+') as f:
            node_degree = json.load(f)
        with open(transit_file_path, 'r+') as f:
            transit_degree = json.load(f)
        with open(AS_hierarchy_file_path, 'r+') as f:
            AS_hierarchy = json.load(f)
        with open(AS_pagerank_file_path, 'r+') as f:
            AS_pagerank = json.load(f)
        with open(AS_Type_file_path, 'r+') as f:
            AS_Type = json.load(f)
        with open(customer_cone_file_path, 'r+') as f:
            AS_customer_cone = json.load(f)
        with open(AS_geolocation_file_path, 'r+') as f:
            AS_geolocation_code = json.load(f)
        with open(high_hierarchy_num_file_path,'r+') as f:
            high_hierarchy_num = json.load(f)
        with open(AS_hegemony_file_path,'r+') as f:
            AS_Hegemony = json.load(f)

        with open(output_file_path, 'w+', newline='') as w:
            csv_writer = csv.writer(w)
            for i in range(len(ASN_list)):
                node_feature_vector = []
                try:
                    node_feature_vector.append(node_degree[ASN_list[i]])
                except:
                    node_feature_vector.append(0)
                try:
                    node_feature_vector.append(transit_degree[ASN_list[i]])
                except:
                    node_feature_vector.append(0)
                try:
                    node_feature_vector.append(AS_hierarchy[ASN_list[i]])
                except:
                    node_feature_vector.append(5)
                try:
                    node_feature_vector.append(AS_pagerank[ASN_list[i]])
                except:
                    node_feature_vector.append(0)
                try:
                    node_feature_vector.append(AS_Type[ASN_list[i]])
                except:
                    node_feature_vector.append(4)
                try:
                    node_feature_vector.append(AS_customer_cone[ASN_list[i]])
                except:
                    node_feature_vector.append(0)
                try:
                    node_feature_vector.append(AS_geolocation_code[ASN_list[i]])
                except:
                    node_feature_vector.append(0)
                try:
                    node_feature_vector.append(high_hierarchy_num[ASN_list[i]])
                except:
                    node_feature_vector.append(0)
                try:
                    node_feature_vector.append(AS_Hegemony[ASN_list[i]])
                except:
                    node_feature_vector.append(0)
                csv_writer.writerow(node_feature_vector)
        print('构建完成！')

    def infer_hidden_links(self):
        '''
        加载Hidden-SAGE模型，对所有可见链接之外的边进行预测
        :return:
        '''

        t1 = time.time()

        print('开始读取特征...')
        for file_name in os.listdir(self.feature_data_folder):
            if file_name.endswith('csv'):  #
                node_features = os.path.join(self.feature_data_folder, file_name)

        # 读取节点特征的 CSV 文件
        with open(node_features, 'r') as csvfile:
            reader = csv.reader(csvfile)
            # 读取节点特征数据
            node_features_data = [list(map(float, row)) for row in reader]
        print('特征读取完毕！')
        print('开始构建图神经网络条件...')
        # 构建图神经网络数据条件
        for file_name in os.listdir(self.progress_data_folder):
            if 'changed_links' in file_name:
                exist_link_file = os.path.join(self.progress_data_folder, file_name)
        src_nodes = []
        dst_nodes = []
        # 加载已存在的链接用于图神经网络
        with open(exist_link_file,'r+') as f:
            for line in f:
                line = line.replace('\n','')
                src_node = line.split('|')[0]
                dst_node = line.split('|')[1]
                src_nodes.append(int(src_node))
                dst_nodes.append(int(dst_node))

        # num_nodes = len(node_features_data)
        num_features = 9  # 节点特征维度  2023年有AS霸权

        # 创建图
        g = dgl.DGLGraph()
        # g.add_nodes(num_nodes)
        g.add_edges(src_nodes, dst_nodes)
        # 将节点特征添加到图 g
        g.ndata['feat'] = torch.tensor(node_features_data)
        # 得到图神经网络产出的embedding
        graph_sage_model = GraphSAGE(num_features, 32, 16)
        node_embedding = graph_sage_model(g, g.ndata['feat'])
        node_embedding = node_embedding.detach()
        print('图神经网络构建完毕！已经得到图神经网络产出embedding！')


        # 加载预测数据
        for file_name in os.listdir(self.progress_data_folder):
            if 'new_links' in file_name:  # 要预测的链接
                new_link_file = os.path.join(self.progress_data_folder, file_name)

        print('正在加载模型...')
        # 从PKL文件中加载模型
        with open('Hidden_link_classifier_model.pth', 'rb') as f:
            Hidden_link_classifier_model = torch.load(f)
        print('模型加载完毕！')
        print(type(Hidden_link_classifier_model))
        print('正在预测...')
        result_file_path = os.path.join(self.result_data_folder, 'hidden_links.txt')
        with open(result_file_path, 'a+') as w:
            with open(new_link_file,'r+') as f:
                for line in f:
                    infer_src_nodes = []
                    infer_dst_nodes = []
                    line = line.replace('\n','')
                    src_node = line.split('|')[0]
                    dst_node = line.split('|')[1]
                    infer_src_nodes.append(int(src_node))
                    infer_dst_nodes.append(int(dst_node))
                    with torch.no_grad():
                        # 获取测试集上的预测结果
                        predict_logits = Hidden_link_classifier_model(infer_src_nodes, infer_dst_nodes, node_embedding)
                        predict_logits = torch.sigmoid(predict_logits)
                        predicted_labels = torch.round(predict_logits)
                        predicted_labels = torch.squeeze(predicted_labels)
                        # print(predicted_labels.item())
                        # print(line+'|'+str(int(predicted_labels.item())))
                        if int(predicted_labels.item()) == 1:
                            print(str(infer_src_nodes[0]) + '|' + str(infer_dst_nodes[0]))
                            w.write(str(infer_src_nodes[0]) + '|' + str(infer_dst_nodes[0]) + '\n')
        print('预测完成！')
        t2 = time.time()
        t3 = t2-t1
        with open('predict_time.txt','w+') as f:
            f.write("共耗时："+str(t3))

        # print('正在预测...')
        # with torch.no_grad():
        #     # 获取测试集上的预测结果
        #     predict_logits = Hidden_link_classifier_model(infer_src_nodes, infer_dst_nodes, node_embedding)
        #     predict_logits = torch.sigmoid(predict_logits)
        #     predicted_labels = torch.round(predict_logits)
        #     predicted_labels = torch.squeeze(predicted_labels)
        # print('预测完毕！')
        #
        # with open(result_file_path, 'a+') as f:
        #     for i in range(len(infer_src_nodes)):
        #         if predicted_labels[i] == 1:
        #             f.write(str(infer_src_nodes[i]) + '|' + str(infer_dst_nodes[i])+ '\n')

    def num2AS(self):
        '''
        将预测后的边映射回AS号，打上标签
        :return:
        '''
        node_2_ASN = defaultdict(str)
        sorted_node_file = ''
        input_file_name = ''
        output_file_name = ''

        with open(sorted_node_file, 'r+') as f:
            for line in f:
                line = line.replace('\n', '')
                ASN = line.split('|')[0]
                node = line.split('|')[1]
                node_2_ASN[node] = ASN

        with open(input_file_name, 'r+') as r:
            with open(output_file_name, 'w+') as w:
                for line in r:
                    line = line.replace('\n', '')
                    node1 = line.split('|')[0]
                    node2 = line.split('|')[1]
                    label = line.split('|')[2]
                    AS1 = node_2_ASN[node1]
                    AS2 = node_2_ASN[node2]
                    link_record = AS1 + '|' + AS2 + '|' + label + '\n'
                    w.write(link_record)

    def add_links(self):
        '''
        将推断出的隐藏链路加入到新链接中并标记
        :return:
        '''
        pass

    def main(self):
        '''

        :return:
        '''
        pass

if __name__ == '__main__':
    source_data_folder = r"D:\projects\python\graduation_design\data\2022\source_data"
    progress_data_folder = r"D:\projects\python\graduation_design\data\2022\progress_data"
    feature_data_folder = r"D:\projects\python\graduation_design\data\2022\features"
    result_data_folder = r"D:\projects\python\graduation_design\data\2022\results"
    ihl = Infer_Hidden_Links(source_data_folder,progress_data_folder,feature_data_folder,result_data_folder)
    # ihl.AS2num()
    # ihl.create_new_links()
    # ihl.create_features()
    ihl.infer_hidden_links()