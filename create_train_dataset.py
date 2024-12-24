from collections import defaultdict
import csv
import json



class Create_Train_Dataset():
    def __init__(self):


        # triple
        self.triple_sorted_node_file = '../train_data/triple_link_samples/triple_sorted_node.txt'
        self.triple_message_links = '../train_data/triple_link_samples/triple_graph.txt'
        self.triple_changed_message_links = '../train_data/triple_link_samples/triple_changed_message_links.txt'
        self.triple_train_sample_file = '../train_data/triple_link_samples/triple_out_samples.txt'
        self.triple_changed_sample_links = '../train_data/triple_link_samples/triple_changed_sample_links.txt'
        self.triple_node_feature = '../train_data/triple_link_samples/triple_node_features.csv'
        # common
        self.common_sorted_node_file = '../train_data/common_link_samples/common_sorted_node.txt'
        self.common_message_links = '../train_data/common_link_samples/common_graph.txt'
        self.common_changed_message_links = '../train_data/common_link_samples/common_changed_message_links.txt'
        self.common_train_sample_file = '../train_data/common_link_samples/common_out_samples.txt'
        self.common_changed_sample_links = '../train_data/common_link_samples/common_changed_sample_links.txt'
        self.common_node_feature = '../train_data/common_link_samples/common_node_features.csv'


    def AS_to_node_graph(self,message_link_file,sorted_node_file,changed_message_links_file):
        ASes = set()
        node_index = defaultdict(int)
        with open(message_link_file,'r+') as f:
            for line in f:
                line = line.replace('\n','')
                try:
                    AS1 = int(line.split('|')[0])
                    AS2 = int(line.split('|')[1])
                    ASes.add(AS1)
                    ASes.add(AS2)
                except:
                    continue
        ASes = sorted(list(ASes))
        with open(sorted_node_file,'w+') as w:
            for i in range(len(ASes)):
                print(ASes[i],i)
                node_index[ASes[i]] = i
                w.write(str(ASes[i])+'\n')
        with open(changed_message_links_file,'w+') as w:
            with open(message_link_file,'r+') as f:
                for line in f:
                    try:
                        AS1 = int(line.split('|')[0])
                        AS2 = int(line.split('|')[1])
                    except:
                        continue
                    AS1_index = str(node_index[AS1])
                    AS2_index = str(node_index[AS2])
                    new_link = AS1_index+'|'+AS2_index
                    w.write(new_link+'\n')

    def AS_to_node_samples(self,sorted_node_file,train_sample_file,sorted_sample_file):
        node_index = defaultdict(int)
        ASes = []
        with open(sorted_node_file,'r+') as f:
            for line in f:
                line = line.replace('\n','')
                ASes.append(int(line))
        ASes = sorted(ASes)
        for i in range(len(ASes)):
            node_index[ASes[i]] = i
        with open(sorted_sample_file, 'w+') as w:
            with open(train_sample_file,'r+') as f:
                for line in f:
                    AS1 = int(line.split('|')[0])
                    AS2 = int(line.split('|')[1])
                    label = line.split('|')[2].replace('\n', '')
                    AS1_index = str(node_index[AS1])
                    AS2_index = str(node_index[AS2])
                    new_link = AS1_index+'|'+AS2_index+'|'+label
                    w.write(new_link+'\n')

    def create_node_feature(self,sorted_node_file,feature_file):
        '''
        node degree|transit degree|AS hierarchy|AS pagerank|AS hegemony|AS Type|AS customer cone|AS geoplocation|AS high hierarchy num

        '''
        node_list = []
        with open(sorted_node_file, 'r+') as f:
            for line in f:
                node = line.replace('\n','')
                node_list.append(node)
        with open('../feature/node_degree.json','r+') as f:
            node_degree = json.load(f)
        with open('../feature/true_transit_degree.json','r+') as f:
            transit_degree = json.load(f)
        with open('../feature/AS_hierarchy.json','r+') as f:
            AS_hierarchy = json.load(f)
        with open('../feature/AS_pagerank.json','r+') as f:
            AS_pagerank = json.load(f)
        with open('../feature/AS_Type.json','r+') as f:
            AS_Type = json.load(f)
        with open('../feature/customer_cone.json','r+') as f:
            AS_customer_cone = json.load(f)
        with open('../feature/AS_geolocation_code.json','r+') as f:
            AS_geolocation_code = json.load(f)
        with open('../feature/high_hierarchy_num.json') as f:
            high_hierarchy_num = json.load(f)

        with open(feature_file, 'w+', newline='') as w:
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
                    node_feature_vector.append(AS_hierarchy[node_list[i]])
                except:
                    node_feature_vector.append(5)
                try:
                    node_feature_vector.append(AS_pagerank[node_list[i]])
                except:
                    node_feature_vector.append(0)
                try:
                    node_feature_vector.append(AS_Type[node_list[i]])
                except:
                    node_feature_vector.append(4)
                try:
                    node_feature_vector.append(AS_customer_cone[node_list[i]])
                except:
                    node_feature_vector.append(0)
                try:
                    node_feature_vector.append(AS_geolocation_code[node_list[i]])
                except:
                    node_feature_vector.append(0)
                try:
                    node_feature_vector.append(high_hierarchy_num[node_list[i]])
                except:
                    node_feature_vector.append(0)
                csv_writer.writerow(node_feature_vector)

    def create_triple_train_data(self):
        self.AS_to_node_graph(self.triple_message_links,self.triple_sorted_node_file,self.triple_changed_message_links)
        self.AS_to_node_samples(self.triple_sorted_node_file,self.triple_train_sample_file,self.triple_changed_sample_links)
        self.create_node_feature(self.triple_sorted_node_file,self.triple_node_feature)

    def create_common_train_data(self):
        self.AS_to_node_graph(self.common_message_links, self.common_sorted_node_file, self.common_changed_message_links)
        self.AS_to_node_samples(self.common_sorted_node_file, self.common_train_sample_file,self.common_changed_sample_links)
        self.create_node_feature(self.common_sorted_node_file, self.common_node_feature)




