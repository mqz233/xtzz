from DataOperator.jsonOperator import jsonOperator as jo
import os
from DataOperator.mysqlOperator import mysqlOperator as mo
from conf.readConfig import readConfig

## data file 高维， 001 动态， init 静态

class newStaticStore():
    def __init__(self,static_root_path,file_name):
        # print("asinoga")
        self.root_data_path = static_root_path
        # data_paths = os.listdir(self.root_data_path)
        # assert len(data_paths)==3
        self.file_name = file_name
        # self.static_data_path = os.path.join(self.root_data_path,'init_001.json')
        self.static_data_path = os.path.join(self.root_data_path, self.file_name)
        self.high_div_data_path = -1
        # self.high_div_data_path = os.path.join(self.root_data_path,'datafile')
        self.normal_dynamic_data_path = -1
        # self.normal_dynamic_data_path = os.path.join(self.root_data_path,'new_dynamic')

        self.static_table_name = 'InitialFrame'

        self.search_tag_init_name = 'Init_1207'#探测
        self.search_tag_last_name = 'Last_1207'

        self.attack_tag_init_name = 'AttackInit'#打击
        self.attack_tag_last_name = 'AttackLast'

        self.repress_tag_init_name = 'RepressInit'  #压制
        self.repress_tag_last_name = 'RepressLast'
        # if len(os.listdir(self.normal_dynamic_data_path))<3:
        #     self.normal_dynamic_data_path = os.path.join(self.normal_dynamic_data_path,'001')

        # print('gga')



    def import_static_datas(self):
        data_content_dict = jo().convertJsonToDict(self.static_data_path)
        keys_list = list(data_content_dict.keys())
        values_list = list(data_content_dict.values())
        len_cols = len(keys_list)
        len_platforms = len(data_content_dict[keys_list[2]])

        ## 在静态表中添加每个平台的数据
        # data_columns= keys_list[2:] ## 所有数据的属性值列表
        # data_values = values_list[2:] ## 所有数值的对应值
        for platform_index in range(0, len_platforms):
            ## 将所有的平台插入数据库
            platform_data = {keys_list[0]: values_list[0], keys_list[1]: values_list[1]}
            # platform_values = [values_list[0],values_list[1]]
            for col_index in range(2, len_cols):
                platform_data[keys_list[col_index]] = values_list[col_index][platform_index]
                # platform_values.append(values_list[col_index][platform_index])
            ## 将组装好的数据插入数据库
            mo().insert_single_data_from_dict(table_name=self.static_table_name, values=platform_data)
########探测数据插入#######
    def import_search_tag_init_datas(self, tag):
        Filelist = []
        for home, dirs, files in os.walk(self.static_data_path):
            # 获得所有文件
            for filename in files:
                Filelist.append(os.path.join(home, filename))
        res = []
        for ff in Filelist:
            if 'init' in ff.split('\\')[-1]:   # 切分出文件名来再判断，可以缩短判断时间
                res.append(ff)


        data_content_dict = jo().convertJsonToDict(res[0])
        keys_list = ['tagid']
        tagid=tag+res[0].split('/')[-1].split('.')[0].split('_')[-1]#self.file_name.split('_')[-1].split('.')[0]
        values_list=[tagid]

        keys_list += list(data_content_dict.keys())
        values_list += list(data_content_dict.values())
        len_cols = len(keys_list)
        len_platforms = len(data_content_dict[keys_list[3]])

        ## 在静态表中添加每个平台的数据
        # data_columns= keys_list[2:] ## 所有数据的属性值列表
        # data_values = values_list[2:] ## 所有数值的对应值
        for platform_index in range(0, len_platforms):
            ## 将所有的平台插入数据库
            platform_data = {keys_list[0]: values_list[0], keys_list[1]: values_list[1],keys_list[2]: values_list[2]}
            # platform_values = [values_list[0],values_list[1]]
            for col_index in range(3, len_cols):
                platform_data[keys_list[col_index]] = values_list[col_index][platform_index]
                # platform_values.append(values_list[col_index][platform_index])
            ## 将组装好的数据插入数据库
            mo().insert_single_tag_data_from_dict(table_name=self.search_tag_init_name, values=platform_data)
        return tagid

    def import_search_tag_last_datas(self, tagid):
        Filelist = []
        for home, dirs, files in os.walk(self.static_data_path):
            # 获得所有文件
            for filename in files:
                Filelist.append(os.path.join(home, filename))
        res = []
        for ff in Filelist:
            if 'last' in ff.split('\\')[-1]:  # 切分出文件名来再判断，可以缩短判断时间
                res.append(ff)

        data_content_dict = jo().convertJsonToDict(res[0]).get('index')

        keys_list = ['tagid']

        values_list=[tagid]

        keys_list += list(data_content_dict.keys())
        values_list += list(data_content_dict.values())
        len_cols = len(keys_list)
        len_platforms = len(data_content_dict[keys_list[3]])

        ## 在静态表中添加每个平台的数据
        # data_columns= keys_list[2:] ## 所有数据的属性值列表
        # data_values = values_list[2:] ## 所有数值的对应值
        for platform_index in range(0, len_platforms):
            ## 将所有的平台插入数据库
            platform_data = {keys_list[0]: values_list[0],
                             keys_list[1]: values_list[1],
                             keys_list[2]: values_list[2],
                             keys_list[8]: values_list[8],
                             keys_list[9]: values_list[9],
                             keys_list[10]: values_list[10]
                             }
            # platform_values = [values_list[0],values_list[1]]
            for col_index in range(3, 8):
                platform_data[keys_list[col_index]] = values_list[col_index][platform_index]
            for col_index in range(11, len_cols):
                platform_data[keys_list[col_index]] = values_list[col_index][platform_index]
                # platform_values.append(values_list[col_index][platform_index])
            ## 将组装好的数据插入数据库
            mo().insert_single_tag_data_from_dict(table_name=self.search_tag_last_name, values=platform_data)

########打击数据插入#######
    def import_attack_tag_init_datas(self, tag):
        Filelist = []
        for home, dirs, files in os.walk(self.static_data_path):
            # 获得所有文件
            for filename in files:
                Filelist.append(os.path.join(home, filename))
        res = []
        for ff in Filelist:
            if 'init' in ff.split('\\')[-1]:   # 切分出文件名来再判断，可以缩短判断时间
                res.append(ff)


        data_content_dict = jo().convertJsonToDict(res[0])
        keys_list = ['tagid']
        tagid=tag+res[0].split('/')[-1].split('.')[0].split('_')[-1]#self.file_name.split('_')[-1].split('.')[0]
        values_list=[tagid]

        keys_list += list(data_content_dict.keys())
        values_list += list(data_content_dict.values())
        len_cols = len(keys_list)
        len_platforms = len(data_content_dict[keys_list[3]])

        ## 在静态表中添加每个平台的数据
        # data_columns= keys_list[2:] ## 所有数据的属性值列表
        # data_values = values_list[2:] ## 所有数值的对应值
        for platform_index in range(0, len_platforms):
            ## 将所有的平台插入数据库
            platform_data = {keys_list[0]: values_list[0], keys_list[1]: values_list[1],keys_list[2]: values_list[2]}
            # platform_values = [values_list[0],values_list[1]]
            for col_index in range(3, len_cols):
                platform_data[keys_list[col_index]] = values_list[col_index][platform_index]
                # platform_values.append(values_list[col_index][platform_index])
            ## 将组装好的数据插入数据库
            mo().insert_single_tag_data_from_dict(table_name=self.attack_tag_init_name, values=platform_data)
        return tagid

    def import_attack_tag_last_datas(self, tagid):
        Filelist = []
        for home, dirs, files in os.walk(self.static_data_path):
            # 获得所有文件
            for filename in files:
                Filelist.append(os.path.join(home, filename))
        res = []
        for ff in Filelist:
            if 'last' in ff.split('\\')[-1]:  # 切分出文件名来再判断，可以缩短判断时间
                res.append(ff)

        data_content_dict = jo().convertJsonToDict(res[0]).get('index')

        keys_list = ['tagid']

        values_list=[tagid]

        keys_list += list(data_content_dict.keys())
        values_list += list(data_content_dict.values())
        len_cols = len(keys_list)
        len_platforms = len(data_content_dict[keys_list[1]])

        ## 在静态表中添加每个平台的数据
        # data_columns= keys_list[2:] ## 所有数据的属性值列表
        # data_values = values_list[2:] ## 所有数值的对应值
        for platform_index in range(0, len_platforms):
            ## 将所有的平台插入数据库
            platform_data = {keys_list[0]: values_list[0],
                             keys_list[2]: values_list[2],
                             keys_list[3]: values_list[3],
                             keys_list[9]: values_list[9],
                             keys_list[10]: values_list[10],
                             keys_list[11]: values_list[11]
                             }
            # platform_values = [values_list[0],values_list[1]]
            platform_data[keys_list[1]] = values_list[1][platform_index]
            for col_index in range(4, 9):
                platform_data[keys_list[col_index]] = values_list[col_index][platform_index]
            for col_index in range(12, len_cols):
                platform_data[keys_list[col_index]] = values_list[col_index][platform_index]
                # platform_values.append(values_list[col_index][platform_index])
            ## 将组装好的数据插入数据库
            mo().insert_single_tag_data_from_dict(table_name=self.attack_tag_last_name, values=platform_data)

########压制数据插入#######
    def import_repress_tag_init_datas(self, tag):
        Filelist = []
        for home, dirs, files in os.walk(self.static_data_path):
            # 获得所有文件
            for filename in files:
                Filelist.append(os.path.join(home, filename))
        res = []
        for ff in Filelist:
            if 'init' in ff.split('\\')[-1]:   # 切分出文件名来再判断，可以缩短判断时间
                res.append(ff)


        data_content_dict = jo().convertJsonToDict(res[0])
        keys_list = ['tagid']
        tagid=tag+res[0].split('/')[-1].split('.')[0].split('_')[-1]#self.file_name.split('_')[-1].split('.')[0]
        values_list=[tagid]

        keys_list += list(data_content_dict.keys())
        values_list += list(data_content_dict.values())
        len_cols = len(keys_list)
        len_platforms = len(data_content_dict[keys_list[3]])

        ## 在静态表中添加每个平台的数据
        # data_columns= keys_list[2:] ## 所有数据的属性值列表
        # data_values = values_list[2:] ## 所有数值的对应值
        for platform_index in range(0, len_platforms):
            ## 将所有的平台插入数据库
            platform_data = {keys_list[0]: values_list[0], keys_list[1]: values_list[1],keys_list[2]: values_list[2]}
            # platform_values = [values_list[0],values_list[1]]
            for col_index in range(3, len_cols):
                platform_data[keys_list[col_index]] = values_list[col_index][platform_index]
                # platform_values.append(values_list[col_index][platform_index])
            ## 将组装好的数据插入数据库
            mo().insert_single_tag_data_from_dict(table_name=self.repress_tag_init_name, values=platform_data)
        return tagid

    def import_repress_tag_last_datas(self, tagid):
        Filelist = []
        for home, dirs, files in os.walk(self.static_data_path):
            # 获得所有文件
            for filename in files:
                Filelist.append(os.path.join(home, filename))
        res = []
        for ff in Filelist:
            if 'last' in ff.split('\\')[-1]:  # 切分出文件名来再判断，可以缩短判断时间
                res.append(ff)

        data_content_dict = jo().convertJsonToDict(res[0]).get('index')

        keys_list = ['tagid']

        values_list=[tagid]

        keys_list += list(data_content_dict.keys())
        values_list += list(data_content_dict.values())
        len_cols = len(keys_list)
        len_platforms = len(data_content_dict[keys_list[1]])

        ## 在静态表中添加每个平台的数据
        # data_columns= keys_list[2:] ## 所有数据的属性值列表
        # data_values = values_list[2:] ## 所有数值的对应值
        for platform_index in range(0, len_platforms):
            ## 将所有的平台插入数据库
            platform_data = {keys_list[0]: values_list[0],
                             keys_list[2]: values_list[2],
                             keys_list[3]: values_list[3],
                             keys_list[9]: values_list[9],
                             keys_list[10]: values_list[10],
                             keys_list[11]: values_list[11]
                             }
            # platform_values = [values_list[0],values_list[1]]
            platform_data[keys_list[1]] = values_list[1][platform_index]
            for col_index in range(4, 9):
                platform_data[keys_list[col_index]] = values_list[col_index][platform_index]
            for col_index in range(12, len_cols):
                platform_data[keys_list[col_index]] = values_list[col_index][platform_index]
                # platform_values.append(values_list[col_index][platform_index])
            ## 将组装好的数据插入数据库
            mo().insert_single_tag_data_from_dict(table_name=self.repress_tag_last_name, values=platform_data)


    def import_tag_frame(self, tag):

        filename = self.static_data_path.split('\\')[-2]
        data_content_dict = jo().convertJsonToDict(self.static_data_path)
        keys_list = ['tagid','warname']
        tagid = tag
        values_list=[tagid,filename]

        keys_list += list(data_content_dict.keys())
        values_list += list(data_content_dict.values())
        len_cols = len(keys_list)
        # len_platforms = len(data_content_dict[keys_list[3]])

        ## 在静态表中添加每个平台的数据
        # data_columns= keys_list[2:] ## 所有数据的属性值列表
        # data_values = values_list[2:] ## 所有数值的对应值

        platform_data = {}
        # platform_values = [values_list[0],values_list[1]]
        for col_index in range(0, len_cols):
            platform_data[keys_list[col_index]] = values_list[col_index]
            # platform_values.append(values_list[col_index][platform_index])
        ## 将组装好的数据插入数据库
        mo().insert_single_tag_data_from_dict(table_name='Frame', values=platform_data)
        return tagid

    def import_tag_index(self, tag):

        filename = self.static_data_path.split('\\')[-2]
        data_content_dict = jo().convertJsonToDict(self.static_data_path)
        keys_list = ['tagid','warname']
        tagid=tag
        values_list=[tagid,filename]

        keys_list += list(data_content_dict.keys())
        values_list += list(data_content_dict.values())
        len_cols = len(keys_list)


        ## 在静态表中添加每个平台的数据
        # data_columns= keys_list[2:] ## 所有数据的属性值列表
        # data_values = values_list[2:] ## 所有数值的对应值

        ## 将所有的平台插入数据库
        platform_data = {}
        # platform_values = [values_list[0],values_list[1]]
        for col_index in range(0,len_cols):
            platform_data[keys_list[col_index]] = values_list[col_index]
            # platform_values.append(values_list[col_index][platform_index])
            ## 将组装好的数据插入数据库
        mo().insert_single_tag_data_from_dict(table_name='Index', values=platform_data)
        return tagid


#前端调用存储单场zz静态数据接口
# newStaticStore(readConfig().getStaticRootPath(),file_name='').import_static_datas()

# 存入所有静态数据
def all_static_store(static_root_path,tagid):#readConfig().getStaticRootPath()
    print(static_root_path)
    path_list = os.listdir(static_root_path) #static_root_path='/root/xtzz/data/new_datas/new_init'
    res = []
    for ff in path_list:
        if 'init' in ff:  # 切分出文件名来再判断，可以缩短判断时间
            res.append(ff)
    newStaticStore(static_root_path,res[0]).import_tag_frame(tagid)
    newStaticStore(static_root_path,'index.json').import_tag_index(tagid)


# tag 探测静态数据存储
def all_search_tag_static_store(static_root_path, tag):#readConfig().getStaticRootPath()
    path_list = os.listdir(static_root_path) #static_root_path='/root/xtzz/data/new_datas/new_init'
    path_list = sorted(path_list,key=lambda keys:[ord(i) for i in keys],reverse=False)
    for file_name in path_list:
        tagid=newStaticStore(static_root_path,file_name).import_search_tag_init_datas(tag)
        newStaticStore(static_root_path, file_name).import_search_tag_last_datas(tagid)

# tag 打击静态数据存储
def all_attack_tag_static_store(static_root_path, tag):#readConfig().getStaticRootPath()
    path_list = os.listdir(static_root_path) #static_root_path='/root/xtzz/data/new_datas/new_init'
    path_list = sorted(path_list,key=lambda keys:[ord(i) for i in keys],reverse=False)
    for file_name in path_list:
        tagid=newStaticStore(static_root_path,file_name).import_attack_tag_init_datas(tag)
        newStaticStore(static_root_path, file_name).import_attack_tag_last_datas(tagid)

# tag 压制静态数据存储
def all_repress_tag_static_store(static_root_path, tag):#readConfig().getStaticRootPath()
    path_list = os.listdir(static_root_path) #static_root_path='/root/xtzz/data/new_datas/new_init'
    path_list = sorted(path_list,key=lambda keys:[ord(i) for i in keys],reverse=False)
    for file_name in path_list:
        tagid=newStaticStore(static_root_path,file_name).import_repress_tag_init_datas(tag)
        newStaticStore(static_root_path, file_name).import_repress_tag_last_datas(tagid)



# all_tag_static_store(readConfig().getStaticRootPath(),'xtzz')
# all_tag_static_store('/root/xtzz/data/评估结果','AAA')

# all_attack_tag_static_store('/root/xtzz/data/打击','Attack')

# all_static_store('../data/output','xtzz')