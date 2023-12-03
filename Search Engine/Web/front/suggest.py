'''
@Project ：Search Engine 
@File    ：suggest.py
@Author  ：Minhao Cao
@Date    ：2023/12/3 16:34 
'''
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import httpx
import json
from flask import render_template, request, jsonify
from . import front

headers = {'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36 Edg/119.0.0.0',
           'referer': 'https://www.baidu.com'}  # 伪装成浏览器访问，通过referer参数伪装来源是百度

@front.route('/suggest')
async def _suggest():
    keywords = request.args.get('keywords')
    if keywords:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f'https://www.baidu.com/sugrec?prod=pc&wd={keywords}', headers=headers) # 利用了百度的搜索建议接口
                return_list = [i['q'] for i in r.json()['g']] # 返回的是json格式的数据，需要用json模块解析

        except Exception as e:
            return_list = []
    else:
        return_list = []
    return jsonify(return_list)