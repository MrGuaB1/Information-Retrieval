'''
@Project ：Search Engine 
@File    ：suggest.py
@Author  ：Minhao Cao
@Date    ：2023/12/3 16:34 
'''
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib
import httpx
import json
from flask import render_template, request, jsonify
from . import front
from Search.search import simple_search
from Search.readData import *;

headers = {'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36 Edg/119.0.0.0',
           'referer': 'https://www.baidu.com'}

@front.route('/suggest')
async def _suggest():
    keywords = request.args.get('keywords', '')
    return_list = []

    # 编码关键词以确保 URL 的安全性
    encoded_keywords = urllib.parse.quote_plus(keywords)

    if keywords:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                baidu_suggest_url = f'https://www.baidu.com/sugrec?prod=pc&wd={encoded_keywords}'
                response = await client.get(baidu_suggest_url)

                response_data = response.json()
                if response_data.get('g'):
                    for suggestion in response_data['g']:
                        if 'q' in suggestion:
                            # 构造包含查询字符串和对应搜索 URL 的字典
                            suggestion_text = suggestion['q']
                            suggestion_url = f"https://www.baidu.com/s?wd={urllib.parse.quote_plus(suggestion_text)}"
                            return_list.append({'text': suggestion_text, 'url': suggestion_url})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify(return_list)

# 实现站内个性化推荐
@front.route('/personalized_recommendation')
def personalized_recommendation():
    search_history = json.loads(request.cookies.get('search_history', '[]'))
    recommendations = []
    for query in search_history:
        results = simple_search(query,[])[:3]
        results = [result[0] for result in results]
        urls = []
        for i in range(len(results)):
            urls.append(allInfo.loc[results[i]]['title'])
        recommendations.extend([[results[i],urls[i]] for i in range(len(results))])
        if len(recommendations) > 8:
            break

    return jsonify(recommendations)
