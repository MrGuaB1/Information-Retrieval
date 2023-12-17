'''
@Project ：Search Engine 
@File    ：webSearch.py
@Author  ：Minhao Cao
@Date    ：2023/12/3 16:36 
'''
import time
from flask import render_template, request, Response
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, RadioField
from wtforms.validators import DataRequired
from . import front
from Search.search import *
from datetime import datetime

class AdvancedSearchForm(FlaskForm):
    all_these_words = StringField('要搜索的内容：', validators=[DataRequired()])
    this_exact_word_or_phrase = StringField('包含全部关键词')
    any_of_these_words = StringField('包含任意关键词')
    none_of_these_words = StringField('不包括关键词')
    site_or_domain = StringField('网站或域名：')
    time_limit = SelectField('时间范围', choices=["任何时间", "一个月内", "一年内"], validators=[DataRequired()])
    is_title_only = RadioField('查询关键词位于', choices=["全部网页", "标题"], validators=[DataRequired()])
    submit = SubmitField("GO Search！")


@front.route('/advanced_search', methods=['GET', 'POST'])
def _advanced_search():
    form = AdvancedSearchForm(is_title_only='全部网页')  # 实例化表单，并且默认搜索全部网页
    if request.method == 'GET':
        if request.args.get('keywords'):
            form.all_these_words.data = request.args.get('keywords')

    if form.validate_on_submit():
        t = time.perf_counter()
        all_these_words = form.all_these_words.data

        if request.cookies.get('search_history'):
            # 从cookie中获取搜索历史
            search_history: list = json.loads(request.cookies.get('search_history'))
        else:
            search_history = []

        if form.is_title_only.data == '标题':
            is_title_only = True
        else:
            is_title_only = False
        try:
            if not is_title_only:  # 搜索全部网页
                result_list = simple_search(all_these_words, search_history)
            else:
                result_list = simple_search(all_these_words, search_history, True)
        except KeyError:
            cost_time = f'{time.perf_counter() - t: .2f}'
            return render_template(r'404.html', keywords=all_these_words, cost_time=cost_time)

        # 拓展结果
        results = expand_results(result_list)

        # 加入额外的搜索限制
        compare_complete = form.this_exact_word_or_phrase.data # 完全匹配
        compare_part = form.any_of_these_words.data # 含有一个词即可
        not_include = form.none_of_these_words.data # 不含有以下词
        time_limit = form.time_limit.data # 时间限制
        website = form.site_or_domain # 检查网站名字

        # 检查发布时间是否符合要求
        if time_limit:
            results = [result for result in results if check_time(result,time_limit)==True]
        # 检查网站来源是否符合要求
        if website.data:
            # 获取input标签中的value字段的值
            domain = str(website).split("value")[-1][2:].split("\"")[0]
            results = [result for result in results if check_website(result,str(domain))==True]
        # 检查是否不含有以下词：
        if not_include:
            results = [result for result in results if check_not_include(result,not_include)==True]
        # 检查是否完全匹配
        if compare_complete:
            results = [result for result in results if check_match_words(result,compare_complete)==True]
        # 检查是否至少含有输入词之一：
        if compare_part:
            results = [result for result in results if check_match_words(result, compare_complete,False) == True]

        cost_time = f'{time.perf_counter() - t: .2f}'
        if len(results) == 0:
            return render_template(r'404.html', keywords=all_these_words, cost_time=cost_time)

        response = Response(render_template(r'result_page.html', keywords=all_these_words, results=results, len_results=len(results), cost_time=cost_time,search_history=search_history))

        if all_these_words not in search_history:
            search_history.append(all_these_words)
        if len(search_history) > 10:
            search_history.pop(0)
        response.set_cookie('search_history', json.dumps(search_history), max_age=60 * 60 * 24 * 30)

        return response

    return render_template(r'advanced_search.html', form=form)


