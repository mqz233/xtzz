import seaborn as sns #需要安包
import numpy as np

def relationship_analysis():
    #数据准备
    data = sns.load_dataset('iris')
    #提取数据
    df = data.iloc[:, :4] #取前四列数据
    #生成相关性矩阵
    result2 = np.corrcoef(df, rowvar=False)
    #存入csv文件
    np.savetxt('mg.csv', result2, delimiter = ',')