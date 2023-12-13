'''
@Project ：Search Engine 
@File    ：search.py
@Author  ：Minhao Cao
@Date    ：2023/12/3 12:27 
'''

from jieba import cut_for_search
from datetime import datetime,timedelta
from Search.readData import *  # 获取所有json和csv数据
import math
import time
import re


# 这个函数主要用于计算输入字符串和历史记录的TF值，文档库的TF值和IDF值我们已经通过前期数据处理得到
def getTF(words, input):
    # 初始化一个词频字典，key为传入的所有的词，value为0
    tf = dict.fromkeys(words, 0)
    for word in input:
        if word in words:
            tf[word] += 1
    for word, count in tf.items():
        tf[word] = math.log10(count + 1)
    return tf

def getTF_IDF(tf, idf):
    tfidf = {}
    for word, count in tf.items():
        tfidf[word] = count * idf[word]
    return tfidf

def getVecLength(key:list)->float:
    """
    :param key: 关键词列表
    :return: 向量长度
    """
    length = 0
    for i in range(len(key)):
        length = length + key[i][1]**2
    return round(math.sqrt(length),2)

# 基础查询，基于向量空间模型
def simple_search(input: str, history: list, onlyTitle: bool = False,num:int = 100):
    """
    :param input: 用户输入的查询字符串
    :param history: 检索历史列表
    :param onlyTitle: 是否启动仅在标题中检索
    :param num：返回结果的数量，默认为100条最相似的
    :return: 一个列表，元素为URL和相似度组成的元组
    """

    # 对输入和历史记录进行分词
    spilt_input = sorted(list(cut_for_search(input)))
    spilt_input = [term for term in spilt_input if term not in [""," "]]
    spilt_history = []
    for i in range(len(history)):
        ls = list(cut_for_search(history[i]))
        ls = [term for term in ls if term not in [""," "]]
        spilt_history.extend(ls)

    # 判断用户需要的搜索模式
    if onlyTitle == True:
        tf_dict = tf_title # readData中读取的json数据
        idf_dict = idf_title
        words = word_set_title
    else:
        tf_dict = tf
        idf_dict = idf
        words = word_set


    tfidf_dict = {}
    for key,value in tf_dict.items():
        tfidf_dict[key] = getTF_IDF(value,idf_dict)
    # 存储关键词的tfidf值，找到num个最大的
    key_tfidf_dict = {}
    for key,value in tfidf_dict.items():
        key_tfidf_dict[key] = sorted(tfidf_dict[key].items(),key=lambda item:item[1],reverse=True)[0:num]
    # 保存关键词字典的key与value
    key_tfidf_dict_keys = list(key_tfidf_dict.keys()) # 即url组成的列表
    key_tfidf_dict_values = list(key_tfidf_dict.values())

    # 用户输入查询的TFIDF
    tf_input = getTF(words,spilt_input)
    tfidf_input = getTF_IDF(tf_input,idf_dict)
    key_input = sorted(tfidf_input.items(),key=lambda item:item[1],reverse=True)[0:num]
    len_key_input = getVecLength(key_input)

    # 历史记录的TFIDF
    tf_history = getTF(words,spilt_history)
    tfidf_history = getTF_IDF(tf_history,idf_dict)
    key_history = sorted(tfidf_history.items(),key=lambda item:item[1],reverse=True)[0:num]
    len_key_history = getVecLength(key_history)

    # 如果词库里没有搜索项，那么返回错误
    if len_key_input == 0:
        raise KeyError

    # 向量空间模型，计算余弦相似度
    key_results = [] # 用于存储余弦相似度
    key_results_index = [] # 记录文档索引
    for i in range(len(key_tfidf_dict_keys)):
        length = 0
        temp_list = key_tfidf_dict_values[i]
        # 遍历每个输入关键词
        for key in key_input:
            if key[1] !=0: # tf-idf值不为0才存在相似度
                # 遍历文档内的每个关键词
                for value in temp_list:
                    if key[0] == value[0]:
                        length = length + key[1]*value[1]
        # 余弦相似度
        sim = round(length/(len_key_input*getVecLength(temp_list)),4)
        key_results.append((key_tfidf_dict_keys[i],sim))
        if sim > 0:
            key_results_index.append(i)
    if len(history) > 0:
        history_results_dict = {}
        for item in key_results_index:
            length = 0
            temp_list = key_tfidf_dict_values[item]
            for _key_history in key_history:
                if _key_history[1] != 0:
                    for value in temp_list:
                        if _key_history[0] == value[0]:
                            length = length + _key_history[1]*value[1]
            sim = round(length/(len_key_history*getVecLength(temp_list)),4)
            history_results_dict[item] = ((key_tfidf_dict_keys[item],sim))

        results = []
        for i in range(len(key_tfidf_dict_keys)):
            if key_results[i][1] == 0:
                pass
            elif j:=history_results_dict.get(i):
                # 设置历史记录的权重为0.1
                results.append((key_results[i][0],key_results[i][1]+j[1]/10))
            else:
                results.append((key_results[i][0],key_results[i][1]))
        results = sorted(results,key=lambda item:item[1],reverse=True)
    # 没有历史记录时，直接利用字典计算余弦相似度即可
    else:
        results = []
        for i in range(len(key_tfidf_dict_keys)):
            results.append((key_results[i][0],key_results[i][1]))
        results = sorted(results, key=lambda item: item[1], reverse=True)

    ls = []
    for res in results:
        if res[1]>0 :
            ls.append((res[0],res[1]))
    return ls

def simple_search_test(input:str,history:list):
    time1 = time.time()
    ret = simple_search(input,history)
    print("在全文中出现的结果：")
    time2 = time.time()
    for item in ret:
        print(item)
    print("在"+str(time2-time1)+"秒时间内响应，返回"+str(len(ret))+"项结果")

    time1 = time.time()
    ret = simple_search(input,history,True)
    time2 = time.time()
    print("仅在标题中出现的结果：")
    for item in ret:
        print(item)
    print("在" + str(time2 - time1) + "秒时间内响应，返回" + str(len(ret)) + "项结果")

# 拓展simple_search的结果
def expand_results(results:list):
    expanded = []
    for res in results:
        url = res[0]
        row = allInfo.loc[url].fillna('')
        title = str(row['title']).replace("_","/")
        dsp = str(row['description'])
        # 计算网页综合得分：0.7*余弦相似度 + 0.3*pagerank值
        score = res[1]*0.7 + 0.3*page_rank.loc[url]['page_rank']
        expanded.append((title,url,dsp,score))
    # 按照综合得分降序排列并返回
    return sorted(expanded,key=lambda item:item[-1],reverse=True)

def expand_results_test(input:str,history:list):
    results = simple_search(input,history,True)
    expanded = expand_results(results)
    for tuple in expanded:
        print(tuple)

# 带有发布时间限制的搜索函数
def check_time(result,limit):
    """
    :param result: simple_search返回结果拓展后的结果的一行
    :param limit：时间限制字符串
    :return: 是否满足要求
    """
    row = allInfo.loc[result[1]]
    if str(row['date_timestamp']) != "nan":
        # 将时间戳转换为datetime
        articleTime = datetime.fromtimestamp(int(row['date_timestamp']))
        res = datetime.now() - articleTime
        if limit == "一周内":
            if res > timedelta(days=7):
                return False
        elif limit == "一个月内":
            if res > timedelta(days=30):
                return False
        elif limit == "一年内":
            if res > timedelta(days=365):
                return False
    if str(row['date_timestamp']) == "nan":
        return False
    return True

def check_time_test(input,limit):
    ret = simple_search(input,[])
    expanded = expand_results(ret)
    print("时间限制添加前的结果，共有"+str(len(expanded))+"条：")
    for item in expanded:
        print(item)
    expanded = [item for item in expanded if check_time(item,limit)==True]
    print("时间限制添加后的结果，共有"+str(len(expanded))+"条：")
    for item in expanded:
        print(item)

# 检查是不是指定的域名或者网站
def check_website(result,name):
    if name not in result[1]:
        return False
    return True

# 检查是否和规定的词匹配，传入一个标志位代表是否进行完全匹配
# 如果不进行完全匹配，那么只要出现一个词就可以判定为True
def check_match_words(result,input,complete=True):
    row = allInfo.loc[result[1]]
    text = f"{row['title']}#{row['description']}#{row['content']}#{row['editor']}"
    ls = re.findall(r'\"(.+?)\"',input)  # 用正则表达式提取双引号中的内容
    for word in ls:
        if word == '#':
            pass
        if word not in text:
            if complete == True:
                return False
        if word in text:
            if complete == False:
                return True
    if complete == True:
        return True
    return False

# 检查是否不含一些词
def check_not_include(result,input):
    row = allInfo.loc[result[1]]
    text = f"{row['title']}#{row['description']}#{row['content']}#{row['editor']}"
    ls = input.replace('\"', '').split('-')
    ls = [word for word in ls if word != '']
    for word in ls:
        if word == '#':
            pass
        if word in text:
            return False
    return True



if __name__ == "__main__":
    #simple_search_test("运动会",['陈雨露'])
    #expand_results_test("运动会",['陈雨露'])
    check_time_test("运动会","一年内")