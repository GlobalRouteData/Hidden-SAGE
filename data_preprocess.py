import os
import platform
import subprocess
import lz4.frame
from crawler import DownloadFile
from calculate_hidden_features import Hidden_Features


class Source_Processor():
    def __init__(self):
        self.year = '2022'
        self.source_data_folder = fr"D:\projects\python\graduation_design\data\{self.year}\source_data"
        self.feature_data_folder = fr'D:\projects\python\graduation_design\data\{self.year}\features'
        self.result_data_folder = fr'D:\projects\python\graduation_design\data\{self.year}\results'

    def unzip_source_data(self,folder_path):
        '''
        在爬取原始数据后将原始数据解压到指定文件夹
        原始数据：
        all-paths
        as2types
        as-org2info
        ppdc-ases

        :return:
        '''

        # 检测操作系统
        system_type = platform.system()

        if system_type == "Windows":
            # 使用 7-Zip 解压所有压缩文件
            # 确保 7-Zip 已经安装并且位于系统的 PATH 中
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".gz") or file_name.endswith(".bz2"):
                    print(file_name)
                    full_path = os.path.join(folder_path, file_name)
                    print(full_path)
                    # 解压到当前目录
                    print(f"正在解压{file_name}至{folder_path}")
                    command = f"7z x {full_path} -o{folder_path}"
                    os.system(command)
                    print("解压成功")
                if file_name.endswith('.lz4'):
                    print(file_name)
                    full_path = os.path.join(folder_path, file_name)
                    print(full_path)
                    with open(full_path, 'rb') as f:
                        compressed_file = f.read()
                    uncompressed_data = lz4.frame.decompress(compressed_file)
                    uncompressed_file_name = file_name.replace('.lz4','')
                    uncompressed_file_path = os.path.join(folder_path,uncompressed_file_name)
                    with open(uncompressed_file_path, 'wb') as w:
                        w.write(uncompressed_data)

            for file_name in os.listdir(folder_path):
                if file_name.endswith(".gz") or file_name.endswith(".bz2") or file_name.endswith(".lz4"):
                    full_path = os.path.join(folder_path, file_name)
                    os.remove(full_path)


        elif system_type == "Linux":
            # 在 Linux 上解压 zip 文件和 tar 文件
            for file_name in os.listdir(folder_path):
                full_path = os.path.join(folder_path, file_name)
                if file_name.endswith(".zip"):
                    result = subprocess.run(
                        ["unzip", full_path, "-d", folder_path], check=True
                    )
                elif file_name.endswith(".tar.gz") or file_name.endswith(".tar.bz2"):
                    result = subprocess.run(
                        ["tar", "-xf", full_path, "-C", folder_path], check=True
                    )


        else:
            print("不支持的操作系统类型")

    def crawl_source_data(self): #####  主要功能函数
        '''
        爬取近10年的all-path数据、客户锥数据、AS类型数据、AS地理位置

        2024/05/08: 由于AS type数据公开太少，直接使用2021年的数据
        :return:
        '''
        all_path_url =f'https://publicdata.caida.org/datasets/2013-asrank-data-supplement/data/{self.year}1201.all-paths.bz2'
        customer_cone_url = f'https://publicdata.caida.org/datasets/2013-asrank-data-supplement/data/{self.year}1201.ppdc-ases.txt.bz2'
        AS_geolocation_url = f'https://publicdata.caida.org/datasets/as-organizations/{self.year}1001.as-org2info.txt.gz'
        AS_hegemony_url = f'https://ihr-archive.iijlab.net/ihr/hegemony/ipv4/global/{self.year}/12/31/ihr_hegemony_ipv4_global_{self.year}-12-31.csv.lz4'
        AS_rel_url = f'https://publicdata.caida.org/datasets/2013-asrank-data-supplement/data/{self.year}1201.as-rel.txt.bz2'
        AS_type_url = 'https://publicdata.caida.org/datasets/as-classification_restricted/20210401.as2types.txt.gz'


        df1 = DownloadFile(customer_cone_url,10,self.source_data_folder)
        df1.main()
        df2 = DownloadFile(AS_geolocation_url,10,self.source_data_folder)
        df2.main()
        df3 = DownloadFile(all_path_url,10,self.source_data_folder)
        df3.main()
        df4 = DownloadFile(AS_hegemony_url,10,self.source_data_folder)
        df4.main()
        df5 = DownloadFile(AS_rel_url, 10, self.source_data_folder)
        df5.main()
        df6 = DownloadFile(AS_type_url,10,self.source_data_folder)
        df6.main()
        self.unzip_source_data(self.source_data_folder)


    def extract_all_links(self):
        '''
        从all-path文件中提取所有可见链接以及AS

        :return:
        '''
        for file_name in os.listdir(self.source_data_folder):
            if file_name.endswith('all-paths'): # 寻找BGP原始文件
                input_file_name = file_name
                output_link_file_name = file_name.replace('all-paths','all-links.txt')
                output_triplet_file_name = file_name.replace('all-paths', 'all-triplets.txt')
                input_file_path = os.path.join(self.source_data_folder,input_file_name)
                output_link_file_path = os.path.join(self.source_data_folder, output_link_file_name)
                output_triplet_file_path = os.path.join(self.source_data_folder, output_triplet_file_name)
                command = f'getAllLinks_Triplets.exe -input={input_file_path} -loutput={output_link_file_path} -toutput={output_triplet_file_path}'
                print('开始提取全部链接')
                os.system(command) # 使用go提取链接
                print('提取结束！')

    def extract_hidden_link_features(self):
        '''
        提取与隐藏链路推断有关的特征
        :return:
        '''
        all_links_file_name = ''
        all_triple_links_file_name = ''
        ppdc_file_name = ''
        as_rel_file_name = ''
        hegemony_file_name = ''
        as_type_file_name = ''
        as_org2info_filename = ''

        # 读取全部原始数据
        for file_name in os.listdir(self.source_data_folder):
            if 'all-links' in file_name:
                all_links_file_name = file_name
            if 'all-triplets' in file_name:
                all_triple_links_file_name = file_name
            if 'ppdc' in file_name:
                ppdc_file_name = file_name
            if 'as-rel' in file_name:
                as_rel_file_name = file_name
            if 'hegemony' in file_name:
                hegemony_file_name = file_name
            if 'as2types' in file_name:
                as_type_file_name = file_name
            if 'as-org2info' in file_name:
                as_org2info_filename = file_name


        # 构建原始数据路径
        all_links_file_path = os.path.join(self.source_data_folder,all_links_file_name)
        all_triple_links_file_path = os.path.join(self.source_data_folder,all_triple_links_file_name)
        ppdc_file_path = os.path.join(self.source_data_folder,ppdc_file_name)
        as_rel_file_path = os.path.join(self.source_data_folder,as_rel_file_name)
        hegemony_file_path = os.path.join(self.source_data_folder,hegemony_file_name)
        as_type_file_path = os.path.join(self.source_data_folder,as_type_file_name)
        as_org2info_path = os.path.join(self.source_data_folder,as_org2info_filename)

        # 构建特征
        hf = Hidden_Features(self.feature_data_folder,all_links_file_path,all_triple_links_file_path,ppdc_file_path,as_rel_file_path,hegemony_file_path,as_type_file_path,as_org2info_path)
        hf.main()

    def main(self):
        self.crawl_source_data()
        self.extract_all_links()
        self.extract_hidden_link_features()

# 功能测试
# if __name__ == '__main__':
#     sp = Source_Processor()
#     sp.crawl_source_data()