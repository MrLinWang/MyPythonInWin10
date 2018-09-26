#!/usr/bin/env python
# coding: utf-8
from bs4 import BeautifulSoup
import requests
import re


##过滤HTML中的标签
#将HTML中标签等信息去掉
def filter_tags(htmlstr):
    #先过滤CDATA
    re_cdata=re.compile('//<!\[CDATA\[[^>]*//\]\]>',re.I) #匹配CDATA
    re_script=re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>',re.I)#Script
    re_style=re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>',re.I)#style
    re_br=re.compile('<br\s*?/?>')#处理换行
    re_h=re.compile('</?\w+[^>]*>')#HTML标签
    re_comment=re.compile('<!--[^>]*-->')#HTML注释
    s=re_cdata.sub('',htmlstr)#去掉CDATA
    s=re_script.sub('',s) #去掉SCRIPT
    s=re_style.sub('',s)#去掉style
    s=re_br.sub('\n',s)#将br转换为换行
    s=re_h.sub('',s) #去掉HTML 标签
    s=re_comment.sub('',s)#去掉HTML注释
    #去掉多余的空行
    blank_line=re.compile('\n+')
    s=blank_line.sub('\n',s)
    return s

#抓取subTarget网页内文章的abstract和keywords
def getAbstractAndKeywords(Target):
    #进入文章页面
    Req = requests.get(Target)
    Html = Req.text
    Html

    bf = BeautifulSoup(Html,"html5lib")
    #抓取title
    title = filter_tags(str(bf.find_all('h1',class_ = 'ArticleTitle')))

    #抓取abstract
    tag = bf.section
    contents = tag.p.contents
    abstract = ''
    for i in range(len(contents)):
        abstract = abstract + str(contents[i])


    abstract = filter_tags(abstract)

    #抓取keywords
    keywords_tag = bf.find_all('span',class_ = 'Keyword')
    keywords = []
    for i in range(len(keywords_tag)):
        keywords.append(filter_tags(str(keywords_tag[i]))[:-1])

    return {"title":title,"abstract":abstract,"keywords":keywords}

url = 'https://link.springer.com'
target = url + '/journal/11192'
req = requests.get(url = target)
html = req.text

bf = BeautifulSoup(html,"html5lib")

a = bf.find_all('li',class_ = 'no-access')

subTarget = a[0].a.get('href')

print(getAbstractAndKeywords(url + subTarget))




