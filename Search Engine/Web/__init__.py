'''
@Project ：Search Engine 
@File    ：__init__.py.py
@Author  ：Minhao Cao
@Date    ：2023/12/3 16:07 
'''
from flask import Flask,request
from flask_bootstrap import Bootstrap5
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect
from Search.readData import *
import jieba
import logging
from logging.handlers import RotatingFileHandler


bootstrap = Bootstrap5()
csrf = CSRFProtect()

# 在每个请求前记录日志

def create_app():
    app = Flask(__name__)

    @app.before_request
    def log_request_info():
        app.logger.info(f"Request: {request.method} {request.url}")

    # 设置日志记录器
    if not app.debug:
        file_handler = RotatingFileHandler('search.log', maxBytes=1024 * 1024, backupCount=1)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Flask application started')

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

