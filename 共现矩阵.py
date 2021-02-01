# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:54:38 2020

@author: leiwei
"""


import os 
import pandas as pd
import numpy  as np 
import jieba
import re
import jieba.analyse
os.chdir('D:/learning/文本聚类/数据集')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer ,TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn



n_features = 3000   #使用关键词数

###数据zhen
def get_csv():
    
    data = list()
    types = 0
    for i in os.listdir('D:/learning/文本聚类/数据集'):   
        print(i)
        df =pd.read_csv(i,header =None,encoding='gbk')
        df.rename(columns={0:"zj",1:'nr'},inplace=True)
        df['fl']=types
        types += 1 
        data.append(df)
    result = pd.concat(data)
    result['nr']=result['nr'].astype(str)
    result['fcjg'] = result['nr'].apply(get_split)
    result['gjc'] = result['fcjg'].apply(get_gjc)
    return result


###分词
def get_split(text):
    seq_list=jieba.cut(text,cut_all=False)
    with open ('D:/learning/中美贸易战评论数据主题建模分析/数据集/chineseStopWords.txt','r',encoding='utf-8') as f:
        stop_worlds = f.read()
    fcjg=list()
    for i in seq_list:       
        i =re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", " ", i) #去标点符号
        if i not in stop_worlds:
            if len(i)==1:
                continue
            elif i == ' ':
                continue           
            else:
                fcjg.append(i)
                
    return  ' '.join(fcjg)
    
###  关键词
def get_gjc(text):
    words = " ".join(jieba.analyse.extract_tags(sentence=text, topK=30, withWeight=False, allowPOS=('n')))  
    return words
           

    
##列表统计
def get_gjc_tj(result) :
    with open ('D:/learning/中美贸易战评论数据主题建模分析/数据集/chineseStopWords.txt','r',encoding='utf-8') as f:
        stop_worlds = f.read()
    gjc_all=list()
    data =list(result['gjc'].values)
    for i in range(len(data)):
        datas =data[i].split(' ')
        for z  in datas:
            gjc_all.append(z)   
    df = pd.DataFrame()
    df['jg'] = gjc_all   ##准备对关键词进行计数
    df_count = df['jg'].value_counts()
    df_count =df_count[0:300]
    df_count = list(df_count.index)
    return data,df_count
 
def get_gcjz(gjcjg,gjc):
  ##共词矩阵构建
    martix = [ ['' for i in range(len(gjc)+1)] for z in range(len(gjc)+1) ]
    martix[0][1:] = gjc
    martix=list(map(list,zip(*martix)))
    martix[0][1:] = gjc
        
    for row in range(1,len(gjc)+1):
        print(row)
        for col in range(1,len(gjc)+1):
            if martix[0][row] ==martix[col][0]:
                martix[col][row]= 0
            else:
                counter =0
                for i in gjcjg: 
                    if martix[0][row] in i and  martix[col][0] in i:
                        counter+=1
                    else:
                        continue
                martix[col][row] = int(counter)
    martix=pd.DataFrame(martix)
    return martix #
 
   
def wb_k_means(data):    
    #文本向量化
    tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                max_features=n_features,
                                stop_words='english',
                                max_df = 0.5,
                                min_df = 10)
    tf = tf_vectorizer.fit_transform(data.fcjg)  ### 词频矩阵
    
    tfdif = TfidfTransformer()  ##调用tf-idf类 实现tf-idf值
    tf_idf =tfdif.fit_transform(tf.toarray()) 
    tf_idf_toarray= tf_idf.toarray()    
    ###文本聚类
    from sklearn.cluster import KMeans
    num_cluseters = 5
    km = KMeans(n_clusters=num_cluseters)
    km.fit(tf_idf_toarray)
    cluster = km.labels_.tolist()
   
    label_pred = km.labels_  ##获取聚类标签
    centroids = km.cluster_centers_ ###获取聚类中心
    inertia = km.inertia_   ##获取聚类准则的总和
    data['cluster'] = cluster
    return centroids


###查看聚类效果
def jl_xg(data):
    data=data.reset_index(drop=True) 
    df = data['fl'].value_counts()
    dfs=pd.crosstab(data.fl, data.cluster, margins=True)
    return dfs
    



def get_lda(data,n_components=5,n_features=6000):
    ####LDA主题模型实现
    # n_features = 3000  #使用关键词数
    n_components=5
    #文本向量化
    tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                    max_features=n_features,
                                    stop_words='english',
                                    max_df = 0.5,
                                    min_df = 10)
    tf = tf_vectorizer.fit_transform(data.fcjg)
 
    
    ##控制主题数
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=50,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
 
    n_top_words = 300  #每个主题显示多少个词
    tf_feature_names = tf_vectorizer.get_feature_names()
    data_list=print_top_words(lda, tf_feature_names, n_top_words)
    data_plot= pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
    pyLDAvis.show(data_plot)
    return tf_feature_names,data_list   

def print_top_words(model, feature_names, n_top_words):
    m=[]
    for topic_idx, topic in enumerate(model.components_):
        
        print("Topic #%d:" % topic_idx)
        datas =" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        m.append(datas)
    print()
    return m



if __name__ =='__main__' :   
    data=get_csv()
    gjcjg,gjc=get_gjc_tj(data)
    martix = get_gcjz(gjcjg,gjc)
    center = wb_k_means(data)   
    data_xg = jl_xg(data)
    data_lda,data_list =get_lda(data)

   




    
    
    
    
    
    
    
    
    
    
    
   
    
    
    