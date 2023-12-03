import json
import os
import pandas as pd

path = "../index/jsons"
rankPath = "../rank"
spiderPath = "../spider"

# 读取index/jsons下的所有数据：
with open(os.path.join(path,'invert_index.json'),'r',encoding='utf-8') as f:
    invert_index = json.load(f)

with open(os.path.join(path,'invert_index_title.json'),'r',encoding='utf-8') as f:
    invert_index_title = json.load(f)

with open(os.path.join(path,'tf-idf.json'),'r',encoding='utf-8') as f:
    tf_idf = json.load(f)

with open(os.path.join(path,'tf-idf_title.json'),'r',encoding='utf-8') as f:
    tf_idf_title = json.load(f)

# 读取词频，即所有出现过的词的TF值
with open(os.path.join(path,"allTF.json"),'r',encoding='utf-8') as f:
    word_frequency = json.load(f)
    word_set = sorted(set(word_frequency.keys()))

with open(os.path.join(path,"allTF_title.json"),'r',encoding='utf-8') as f:
    word_frequency_title = json.load(f)
    word_set_title = sorted(set(word_frequency_title.keys()))

with open(os.path.join(path,"tf.json"),'r',encoding='utf-8') as f:
    tf = json.load(f)

with open(os.path.join(path,"tf_title.json"),'r',encoding='utf-8') as f:
    tf_title = json.load(f)

with open(os.path.join(path,"idf.json"),'r',encoding='utf-8') as f:
    idf = json.load(f)

with open(os.path.join(path,"idf_title.json"),'r',encoding='utf-8') as f:
    idf_title = json.load(f)


# 读取csv文件：
page_rank = pd.read_csv(os.path.join(rankPath,"page_rank.csv"),encoding='utf-8-sig',index_col=0)
allInfo = pd.read_csv(os.path.join(spiderPath,"allInfo.csv"),encoding='utf-8',index_col=0)