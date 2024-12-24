import json
import os.path

import networkx as nx
from collections import defaultdict
import csv

class Hidden_Features():
    '''
    用于计算AS节点特征的类
    AS节点特征包括：
    节点度
    过境度
    AS层级
    AS pagerank
    AS类型
    AS客户锥
    AS地理位置
    邻居高层AS数量
    '''

    def __init__(self,feature_folder,all_links,all_triplets,ppdc,AS_rel,hege_csv,AS_type,org2info):
        # input 原始数据
        self.all_links = all_links
        self.all_triple_links = all_triplets
        self.ppdc = ppdc
        self.AS_rel = AS_rel
        self.unzipped_csv = hege_csv
        self.AS_type = AS_type
        self.org2info = org2info
        # output 输出特征文件的路径
        self.output_pagerank_path = os.path.join(feature_folder,'AS_pagerank.json')
        self.output_degree_path = os.path.join(feature_folder,'degree.json')
        self.output_transit_degree_path = os.path.join(feature_folder,'transit_degree.json')
        self.output_AS_hierarchy_path = os.path.join(feature_folder,'AS_hierarchy.json')
        self.output_AS_type_path = os.path.join(feature_folder,'AS_type.json')
        self.output_customer_cone_path = os.path.join(feature_folder,'customer_cone.json')
        self.output_geolocation_path = os.path.join(feature_folder,'AS_geolocation_code.json')
        self.output_high_hierarchy_neighbour_path = os.path.join(feature_folder,'high_hierarchy_neighbour.json')
        self.output_AS_hegemony_path = os.path.join(feature_folder,'AS_hegemony.json')

    def cal_pagerank_node_degree(self):
        print('Calculating pagerank and degree...')
        node_set = set()
        edge_set = []
        with open(self.all_links,'r+') as f:
            for line in f:
                link = line.replace('\n','')
                AS1 = link.split('|')[0]
                AS2 = link.split('|')[1]
                node_set.add(AS1)
                node_set.add(AS2)
                edge_set.append((AS1,AS2))
        G = nx.Graph()
        G.add_nodes_from(list(node_set))
        G.add_edges_from(edge_set)
        pagerank_values = nx.pagerank(G)
        nodes_degree = dict(nx.degree(G))
        with open(self.output_pagerank_path,'w+') as w:
            json.dump(pagerank_values,w)
        with open(self.output_degree_path,'w+') as w:
            json.dump(nodes_degree,w)
        print('Finished calculating pagerank and degree!')


    def cal_transit_degree(self):
        print('Calculating transit degree...')
        transit_dict = defaultdict(set)
        transit_degree = defaultdict(int)
        with open(self.output_degree_path,'r+') as f:
            node_degree = json.load(f)
        with open(self.all_triple_links,'r+') as f:
            for line in f:
                line = line.replace('\n','')
                parts = line.split('|')
                for i in range(1,len(parts)-1):
                    transit_dict[parts[i]].add(parts[i-1])
                    transit_dict[parts[i]].add(parts[i +1])
        for key in transit_dict.keys():
            transit_degree[key] = len(transit_dict[key])
        for key in node_degree.keys():
            if key not in transit_degree.keys():
                transit_degree[key] = 0
        with open(self.output_transit_degree_path,'w+') as w:
            json.dump(transit_degree,w)
        print('Finished calculating transit degree!')

    def cal_AS_hierarchy(self):
        '''
        1:clique AS
        2:高层 AS
        3:底层 AS
        4:stub AS
        '''
        print('Calculating AS hierarchy...')
        hierarchy_dict = defaultdict(int)
        direct_neighbour_set = set()
        clique_as = ["174", "209", "286", "701", "1239", "1299", "2828", "2914", "3257", "3320", "3356", "5511", "6453",
                     "6461", "6762", "7018", "12956"]
        with open(self.AS_rel, 'r+') as f:
            for line in f:
                if '#' in line:
                    continue
                AS1 = line.split('|')[0]
                AS2 = line.split('|')[1]
                AS_rel = line.split('|')[2]
                if AS_rel == '-1':
                    if AS1 in clique_as:
                        direct_neighbour_set.add(AS2)
        with open(self.output_degree_path, 'r+') as f:
            node_degree = json.load(f)
        with open(self.output_transit_degree_path, 'r+') as f:
            for AS in clique_as:
                hierarchy_dict[AS] = 1
            transit_degree_dict = json.load(f)
            for key in transit_degree_dict.keys():
                if key not in clique_as:
                    if transit_degree_dict[key] == 0:
                        hierarchy_dict[key] = 4
                    else:
                        if node_degree[key] > 100 and key in direct_neighbour_set:
                            hierarchy_dict[key] = 2
                        else:
                            hierarchy_dict[key] = 3
            with open(self.output_AS_hierarchy_path, 'w+') as r:
                json.dump(hierarchy_dict, r)
            print('Finished calculating AS hierarchy!')

    def cal_AS_type(self):
        '''
        0:Transit/Access
        1:Content
        2:Enterpise
        '''
        print('Calculating AS type...')
        type_dict = defaultdict(int)
        with open(self.AS_type, 'r+') as f:
            for line in f:
                if '#' in line:
                    continue
                ASN = line.split('|')[0]
                AS_TYPE = line.split('|')[-1].replace('\n', '')
                if AS_TYPE == 'Transit/Access':
                    type_dict[ASN] = 0
                if AS_TYPE == 'Content':
                    type_dict[ASN] = 1
                if AS_TYPE == 'Enterpise':
                    type_dict[ASN] = 2
            with open(self.output_AS_type_path, 'w+') as r:
                json.dump(type_dict, r)
        print('Finished calculating AS type!')
    def create_customer_cone(self):
        print('Calculating customer cone...')
        customer_cone_amount_dict = defaultdict(int)
        with open(self.ppdc,'r+') as f:
            for line in f:
                if '#' in line:
                    continue
                line = line.replace('\n','')
                customer_list = line.split(' ')
                target_AS = customer_list[0]
                customer_cone_amount = len(customer_list)-1
                customer_cone_amount_dict[target_AS] = customer_cone_amount
        with open(self.output_customer_cone_path,'w+') as w:
            json.dump(customer_cone_amount_dict,w)
        print('Finished calculating customer cone!')

    def create_AS_geolocation(self):
        print('Calculating AS geolocation...')
        org_country_dict = defaultdict(str)
        AS_country_dict = defaultdict(str)
        AS_country_code = defaultdict(int)
        with open(self.org2info, 'r+', encoding='utf-8') as f:
            for line in f:
                if '#' in line:
                    continue
                line = line.replace('\n', '')
                parts = line.split('|')
                if len(parts) == 5:
                    org_id = parts[0]
                    country_name = parts[3]
                    org_country_dict[org_id] = country_name
                if len(parts) == 6:
                    AS_num = parts[0]
                    org_id = parts[3]
                    AS_country_dict[AS_num] = org_country_dict[org_id]
        for key in AS_country_dict.keys():
            sum = 0
            for char in AS_country_dict[key]:
                sum += ord(char)
            AS_country_code[key] = sum
        with open(self.output_geolocation_path, 'w+') as w:
            json.dump(AS_country_code, w)
        print('Finished calculating AS geolocation!')

    def create_high_tier_neighbour(self):

        # 假设你已经构建了一个图，命名为 G
        # G = nx.Graph()  # 这里需要根据你的实际情况创建图

        # 假设你有一个特定的链接
        # specific_link = ('123', '234')  # 这里需要根据你的实际情况提供特定的链接
        #
        # # 获取特定链接的邻居节点列表
        # neighbors = list(G.neighbors(specific_link))
        #
        # print("Neighbors of {}: {}".format(specific_link, neighbors))
        print('Calculating high hierarchy neighbours...')
        node_set = set()
        edge_set = []
        high_hierarchy_dict = defaultdict(int)
        with open(self.output_AS_hierarchy_path,'r+') as f:
            AS_hierarchy = json.load(f)
        with open(self.all_links, 'r+') as f:
            for line in f:
                link = line.replace('\n', '')
                AS1 = link.split('|')[0]
                AS2 = link.split('|')[1]
                node_set.add(AS1)
                node_set.add(AS2)
                edge_set.append((AS1, AS2))
        G = nx.Graph()
        G.add_nodes_from(list(node_set))
        G.add_edges_from(edge_set)
        n = 0
        total = len(list(node_set))
        exist_high = False
        for node in list(node_set):
            for neighbour in list(G.neighbors(node)):
                if AS_hierarchy[neighbour] == 1 or AS_hierarchy[neighbour] == 2:
                    high_hierarchy_dict[node] += 1
                    exist_high = True
            if exist_high == False:
                high_hierarchy_dict[node] = 0
            exist_high = False

            n += 1
            print(f'finished: {n}/{total}')
        # print(len(high_hierarchy_dict.keys()))
        with open(self.output_high_hierarchy_neighbour_path,'w+') as w:
            json.dump(high_hierarchy_dict,w)
        print('Finished calculating high hierarchy neighbours!')
    def create_AS_Hegemony(self):
        print('Calculating AS hegemony...')
        AS_hegemony_dict = defaultdict(float)
        with open(self.unzipped_csv,'r+') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)
            for row in csv_reader:
                ASN = row[2]
                hegemony = row[3]
                AS_hegemony_dict[ASN] = hegemony
        with open(self.output_AS_hegemony_path,'w+') as w:
            json.dump(AS_hegemony_dict,w)
        print('Finished calculating AS hegemony!')


    def main(self):
        self.cal_pagerank_node_degree()
        self.cal_transit_degree()
        self.cal_AS_hierarchy()
        self.cal_AS_type()
        self.create_customer_cone()
        self.create_AS_geolocation()
        self.create_high_tier_neighbour()
        self.create_AS_Hegemony()







