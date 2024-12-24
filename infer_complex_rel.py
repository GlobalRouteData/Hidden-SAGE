from collections import defaultdict
import networkx as nx
import json
import pickle
import os
import csv

class Infer_Complex_Relationships():

    def __init__(self,feature_data_folder,all_path_file):
        self.feature_data_folder = feature_data_folder
        self.all_path_file = all_path_file

    def extract_complex_rels_features(self):
        '''
        提取复杂商业关系推断使用的特征
        2024/05/24：其实也就多一个distance to clique
        :return:
        '''
        for file_name in os.listdir(self.feature_data_folder):
            if 'degree.json' in file_name:
                degree_file_path = os.path.join(self.feature_data_folder,file_name)

        path_list = []
        distance_dict = defaultdict(list)
        clique_distance = defaultdict(int)
        clique_AS = ["174", "209", "286", "701", "1239", "1299", "2828", "2914", "3257", "3320", "3356", "5511", "6453",
                     "6461", "6762", "7018", "12956"]

        with open(self.all_path_file, 'r+') as f:
            for line in f:
                ASpath = line.split(' ')[1]
                path_list.append(ASpath)
        print('ASpath list created!')
        G = nx.Graph()
        for path in path_list:
            nodes = path.split('|')
            print(path)
            for i in range(len(nodes) - 1):
                G.add_edge(nodes[i], nodes[i + 1])
        print('Graph created!')
        with open(degree_file_path, 'r+') as r:
            node_dict = json.load(r)
        for key in node_dict.keys():
            for clique in clique_AS:
                try:
                    distance = nx.shortest_path_length(G, key, clique)
                    distance_dict[key].append(distance)
                except nx.NetworkXNoPath:
                    continue
        print('Calculate finish!')
        for key in distance_dict.keys():
            value = sum(distance_dict[key]) / len(distance_dict[key])
            clique_distance[key] = value
        d2c_file_path = os.path.join(self.feature_data_folder,'distance_to_clique.json')
        with open(d2c_file_path, 'w+') as w:
            json.dump(clique_distance, w)
        print('All finish!')

    def create_complex_rel_feature_csv(self):
        '''
           node degree|transit degree|distance to clique|AS type|AS hierarchy|AS geolocation|AS pagerank|AS_high_hierarchy_num

           '''
        node_list = []
        with open(sorted_node_file, 'r+') as f:
            for line in f:
                node = line.replace('\n', '')
                node_list.append(node)
        with open('../features/node_degree.json', 'r+') as f:
            node_degree = json.load(f)
        with open('../features/true_transit_degree.json', 'r+') as f:
            transit_degree = json.load(f)
        with open('../features/distance_to_clique.json', 'r+') as f:
            distance_to_clique = json.load(f)
        with open('../features/AS_Type.json', 'r+') as f:
            AS_Type = json.load(f)
        with open('../features/AS_hierarchy.json', 'r+') as f:
            AS_hierarchy = json.load(f)
        with open('../features/AS_geolocation_code.json', 'r+') as f:
            AS_geolocation_code = json.load(f)
        with open('../features/AS_pagerank.json', 'r+') as f:
            AS_pagerank = json.load(f)
        with open('../features/high_hierarchy_num.json', 'r+') as f:
            high_hierarchy_num = json.load(f)
        with open('node_features.csv', 'w+', newline='') as w:
            csv_writer = csv.writer(w)
            for i in range(len(node_list)):
                node_feature_vector = []
                try:
                    node_feature_vector.append(node_degree[node_list[i]])
                except:
                    node_feature_vector.append(0)
                try:
                    node_feature_vector.append(transit_degree[node_list[i]])
                except:
                    node_feature_vector.append(0)
                try:
                    node_feature_vector.append(distance_to_clique[node_list[i]])
                except:
                    node_feature_vector.append(0)
                try:
                    node_feature_vector.append(AS_Type[node_list[i]])
                except:
                    node_feature_vector.append(4)
                try:
                    node_feature_vector.append(AS_hierarchy[node_list[i]])
                except:
                    node_feature_vector.append(5)
                try:
                    node_feature_vector.append(AS_geolocation_code[node_list[i]])
                except:
                    node_feature_vector.append(0)
                try:
                    node_feature_vector.append(AS_pagerank[node_list[i]])
                except:
                    node_feature_vector.append(0)
                try:
                    node_feature_vector.append(high_hierarchy_num[node_list[i]])
                except:
                    node_feature_vector.append(0)
                csv_writer.writerow(node_feature_vector)

        with open('node_features.csv', 'r+') as f:
            reader = csv.reader(f)
            # 读取节点特征数据
            node_features_data = [list(map(float, row)) for row in reader]
        # print(node_features_data[0]+node_features_data[1])
        with open('sorted_samples.txt', 'r+') as f:
            with open('tree_data.csv', 'w+') as w:
                csv_writer = csv.writer(w)
                for line in f:
                    line = line.replace('\n', '')
                    print(line)
                    parts = line.split('|')
                    AS1 = int(parts[0])
                    AS2 = int(parts[1])
                    label = parts[2]
                    exp_data = node_features_data[AS1] + node_features_data[AS2]
                    exp_data.append(label)
                    csv_writer.writerow(exp_data)

    def infer_complex_rels(self):
        '''
        拼接特征向量并加载决策树模型预测
        :return:
        '''
        complex_rel_model = ''
        with open('./random_forest_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        dataset = pd.read_csv('../train_data/tree_data9.csv', header=None,
                              names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                                     'q'])
        print(np.isnan(dataset).any())
        dataset.dropna(inplace=True)
        y_pred = xgbcf.predict(X_test)












