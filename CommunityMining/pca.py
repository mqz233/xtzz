import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataOperator.jsonOperator import jsonOperator as jo
datas_path="D:\pycharm\project\项目\东大EXA_6v4输出数据\json_output_5055555\R\R_00002.json"

if __name__ == '__main__':
    datas = jo().convertJsonToDict(datas_path)
    # print(datas)
    # print(datas.get("platforms"))
    platforms_list=datas.get("platforms")
    pd.set_option("display.max_columns",None)
    platforms_pd=pd.DataFrame(platforms_list)
    # print(platforms_pd)
    platforms=platforms_pd.copy()

    platforms.master=platforms.master.replace({True:1,False:0}) # replace
    platforms.radar=platforms.radar.replace({True:1,False:0})
    platforms.drop(columns=["id","svv","type","posx","posy","posz"],inplace=True)
    # print(platforms)
    platforms.describe()
    def norm_(x):
        xmean=np.mean(x,0)
        std=np.std(x,0)
        return (x-xmean)/std
    data_=norm_(platforms)
    data_=data_.fillna(0)
    # print(platforms_)
    ew,ev=np.linalg.eig(np.cov(data_.T))
    ew_order=np.argsort(ew)[::-1]
    ew_sort=ew[ew_order]
    ev_sort=ev[:,ew_order]
    df=pd.DataFrame(ew_sort)
    df.plot(kind="bar")
    plt.show()


    V=ev_sort[:,:2]
    X_new=data_.dot(V)
    sc=plt.scatter(X_new.iloc[:,0],X_new.iloc[:,1],s=5,c=platforms_pd.svv,cmap=plt.cm.coolwarm)
    plt.xlabel('PC0')
    plt.ylabel("PC1")
    plt.colorbar(sc)
    plt.show()

