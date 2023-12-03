'''
@Project ：Search Engine 
@File    ：snapshot.py
@Author  ：Minhao Cao
@Date    ：2023/12/3 16:33 
'''
import json

from flask import render_template, request

from . import front
from Search.readData import *


@front.route('/snapshot')
def _snapshot():
    if url := request.args.get('url'):
        title = allInfo.loc[url]['title']
        with open(rf'./tools/pages/{title}.html',encoding='utf-8') as f:
            snapshot = f.read()
        # 向前端以网页的形式返回快照
        return render_template(r'snapshot.html', snapshot=snapshot)
    else:
        return "不合法的参数"