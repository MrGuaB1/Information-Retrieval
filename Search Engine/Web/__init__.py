'''
@Project ：Search Engine 
@File    ：__init__.py.py
@Author  ：Minhao Cao
@Date    ：2023/12/3 16:07 
'''
from flask import Flask
from flask_bootstrap import Bootstrap5
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect
from Search.readData import *
import jieba


bootstrap = Bootstrap5()
csrf = CSRFProtect()


def create_app():
    app = Flask(__name__)

    bootstrap.init_app(app)
    csrf.init_app(app)
    CORS(app, supports_credentials=True)

    app.config['BOOTSTRAP_SERVE_LOCAL'] = True
    app.config["SECRET_KEY"] = '81a96b4d1a8b3bd368dcfa7cfe331b80'
    app.config["WTF_CSRF_SECRET_KEY "] = "secret key 114514"

    jieba.lcut('初始化分词器')

    from Web.front import front as front_blueprint
    app.register_blueprint(front_blueprint)

    return app