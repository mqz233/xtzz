import configparser
import os


class readConfig:
    """
    配置文件类，用于读取所有配置文件并封装成函数
    """
    _instance = None

    ## 单例模式，所有配置共用同一个实例化对象
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
        self.config.read(self.config_path, encoding='UTF-8')
        # self.config.read("/conf/config.ini", encoding='UTF-8')
        self.sections = self.config.sections()
        # return self.instance

    def getSections(self):
        """
        获取配置文件中所有的节，即默认参数、请求与基站三部分
        :return: 配置中所有用[]引用的内容
        """
        return self.sections

    ## 查看 mysql 域中的信息
    def getMysqlItems(self):
        '''
        :return: list of default keys: list
        : list[key][1]
        '''
        items = self.config.items('mysql')
        return dict(items)

    def getMysqlConfig(self):
        mysqlConfig = self.getMysqlItems()
        mysqlConfig['port'] = int(mysqlConfig['port'])
        return mysqlConfig

    ## 查看 influxdb 域中的信息
    def getInfluxdbItems(self):
        '''
        :return: list of default keys: list
        : list[key][1]
        '''
        items = self.config.items('InfluxDB')
        return dict(items)

    def getInfluxdbHost(self):
        return self.getInfluxdbItems()['host']

    def getInfluxdbPort(self):
        return self.getInfluxdbItems()['port']
    def getInfluxdbBb(self):
        return self.getInfluxdbItems()['db']

    ## 获取 data 域中的信息
    def getDataItems(self):
        items = self.config.items('data')
        return dict(items)

    def getDataRootDirPath(self):
        return self.getDataItems()['root_dir']

    def getPreprocessedDataPath(self):
        return self.getDataItems()['preprocessed_data']

    def getPreprocessedStaticDataPath(self):
        return self.getDataItems()['preprocessed_static_data']

    def getPreprocessedDynamicDataPath(self):
        return self.getDataItems()['preprocessed_dynamic_data']

    def getPreprocessedSvvDataPath(self):
        return self.getDataItems()['preprocessed_svv_data']

    def getPreprocessedSamplesDataPath(self):
        return self.getDataItems()['preprocessed_samples_data']

    def getDatasetPath(self):
        return self.getDataItems()['dataset']

    def getStaticDatasetPath(self):
        return self.getDataItems()['static_dataset']

    def getImagePath(self):
        items = self.config.items('image')
        return dict(items)['image_path']

    def getStaticRootPath(self):
        return self.getDataItems()['static_root_path']
    def getDynamicRootPath(self):
        return self.getDataItems()['dynamic_root_path']
    def getZipRootPath(self):
        return self.getDataItems()['zip_root_path']
    def getUnZipRootPath(self):
        return self.getDataItems()['unzip_root_path']
    def getPredictRootPath(self):
        return self.getDataItems()['predict_root_path']
    def getResultPredictRootPath(self):
        return self.getDataItems()['result_predict_root_path']
# print(str(readConfig().getInfluxdbHost()))