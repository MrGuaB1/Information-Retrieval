'''
@Project ：Search Engine 
@File    ：run.py
@Author  ：Minhao Cao
@Date    ：2023/12/3 17:10 
'''
from Web import create_app
app = create_app()

if __name__ == '__main__':
    app.run(debug=True,host="127.0.0.1",port=5000)