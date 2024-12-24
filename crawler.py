import logging
import os
import time
from threading import Thread
import httpx
from tqdm import tqdm
import argparse


'''
这个脚本的功能是下载特定收集器的指定月份的BGP信息
具体使用方法是在终端中利用命令行输入必要参数，然后就可以开始下载
'''
parser = argparse.ArgumentParser()
parser.add_argument('-m','--more',type=str,default=None, metavar='', help='set a begin point of download set.')
args = parser.parse_args()


def logger_config(log_path, logging_name):
    '''
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    '''
    '''
    logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    '''
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出ERROR及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.INFO)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
    handler.setFormatter(formatter)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    return logger

thread_logger = logger_config(log_path='thread_log.txt', logging_name="thread_errs")
download_logger = logger_config(log_path='download_log.txt', logging_name="download_infos")


# 下载类
'''
使用方法：
classname  = DownloadFile(url, thread_num)
classname.main()
拿去下点儿别的什么也可以
'''
class DownloadFile(object):
    def __init__(self, download_url, thread_num,data_folder):
        """
        :param download_url: 文件下载连接
        :param thread_num: 开辟线程数量
        """
        self.download_url = download_url
        self.thread_num = thread_num
        self.file_size = None
        self.cut_size = None
        self.tqdm_obj = None
        self.thread_list = []

        # 设置下载文件存放路径
        # self.data_folder = 'D:\IDEs\pycharmProjects\graduation_design\data\source_data'
        self.data_folder = data_folder
        self.file_path = os.path.join(self.data_folder, download_url.split('/')[-1])



    def downloader(self, etag, thread_index, start_index, stop_index, retry=False, retry_time=0):
        sub_path_file = "{}_{}".format(self.file_path, thread_index)
        if os.path.exists(sub_path_file):
            temp_size = os.path.getsize(sub_path_file)  # 本地已经下载的文件大小
            if not retry:
                self.tqdm_obj.update(temp_size)  # 更新下载进度条
        else:
            temp_size = 0
        if stop_index == '-': stop_index = ""
        headers = {'Range': 'bytes={}-{}'.format(start_index + temp_size, stop_index),
                   'ETag': etag, 'if-Range': etag,
                   }
        down_file = open(sub_path_file, 'ab')
        try:
            with httpx.stream("GET", self.download_url, headers=headers) as response:
                num_bytes_downloaded = response.num_bytes_downloaded
                for chunk in response.iter_bytes():
                    if chunk:
                        down_file.write(chunk)
                        self.tqdm_obj.update(response.num_bytes_downloaded - num_bytes_downloaded)
                        num_bytes_downloaded = response.num_bytes_downloaded
        except Exception as e:
            if retry_time < 20:
                retry_time +=1
                print("Thread-{}:请求超时,第{}次重试\n报错信息:{}".format(thread_index, retry_time,e))
                self.downloader(etag, thread_index, start_index, stop_index, retry=True, retry_time=retry_time)
            else:
                print('因为线程超时重试次数太多，将未完成的文件写入日志中...')
                thread_logger.error("因线程{}超时下载未完成".format(thread_index))
        finally:
            down_file.close()
        return

    def get_file_size(self):
        """
        获取预下载文件大小和文件etag
        :return:
        """
        with httpx.stream("HEAD", self.download_url) as response2:
            etag = ''
            total_size = int(response2.headers["Content-Length"])
            for tltle in response2.headers.raw:
                if tltle[0].decode() == "ETag":
                    etag = tltle[1].decode()
                    break
        return total_size, etag

    def cutting(self):
        """
        切割成若干份
        :param file_size: 下载文件大小
        :param thread_num: 线程数量
        :return:
        """
        cut_info = {}
        cut_size = self.file_size // self.thread_num
        for num in range(1, self.thread_num + 1):
            if num != 1:
                cut_info[num] = [cut_size, cut_size * (num - 1) + 1, cut_size * num]
            else:
                cut_info[num] = [cut_size, cut_size * (num - 1), cut_size * num]
            if num == self.thread_num:
                cut_info[num][2] = '-'
        return cut_info, cut_size

    def write_file(self):
        """
        合并分段下载的文件
        :param file_path:
        :return:
        """
        if os.path.exists(self.file_path):
            return
        with open(self.file_path, 'ab') as f_count:
            for thread_index in range(1, self.thread_num + 1):
                with open("{}_{}".format(self.file_path, thread_index), 'rb') as sub_write:
                    f_count.write(sub_write.read())
                # 合并完成删除子文件
                os.remove("{}_{}".format(self.file_path, thread_index))
        return

    def create_thread(self, etag, cut_info):
        """
        开辟多线程下载
        :param file_path: 文件存储路径
        :param etag: headers校验
        :param cut_info:
        :return:
        """

        for thread_index in range(1, self.thread_num + 1):
            thread = Thread(target=self.downloader,
                            args=(etag, thread_index, cut_info[thread_index][1], cut_info[thread_index][2]))

            thread.setName('Thread-{}'.format(thread_index))
            thread.setDaemon(True)
            thread.start()
            self.thread_list.append(thread)

        for thread in self.thread_list:
            thread.join()
        return

    def check_thread_status(self):
        """
        查询线程状态。
        :return:
        """
        while True:
            for thread in self.thread_list:
                thread_name = thread.getName()
                if not thread.isAlive():
                    print("{}:已停止".format(thread_name))
            time.sleep(1)

    def create_data(self):
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        return

    def main(self):
        # 平分几份
        self.create_data()
        self.file_size, etag = self.get_file_size()
        # 按线程数量均匀切割下载文件
        cut_info, self.cut_size = self.cutting()
        # 下载文件名称
        # 创建下载进度条
        self.tqdm_obj = tqdm(total=self.file_size, unit_scale=True, desc=self.file_path.split('/')[-1],
                             unit_divisor=1024,
                             unit="B")
        # 开始多线程下载
        self.create_thread(etag, cut_info)
        # 合并多线程下载文件
        self.write_file()
        return



'''
-m 参数格式不知道怎么写的时候看下载日志  日志里的字符串长啥样就把他们用‘/‘连起来，例如
route-views.widebgpdata2022.10RIBSrib.20221025.2200.bz2下载完成
-m 里就写 -m route-views.wide/bgpdata/2022.10/RIBS/rib.20221025.2200.bz2

'''
