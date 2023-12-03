'''
@Project ：Search Engine 
@File    ：__init__.py.py
@Author  ：Minhao Cao
@Date    ：2023/12/3 16:14 
'''
from flask import Blueprint
from Web import csrf
csrf = csrf
front = Blueprint("front", __name__)
from . import index,result,webSearch,snapshot,suggest