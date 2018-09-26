#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

#抓取Target网页内文章的title,abstract和keywords
def getPaperInfo(Target):
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


# In[2]:


#进入首页
url = 'https://link.springer.com'
target = url + '/journal/11192'
req = requests.get(url = target)
html = req.text

bf = BeautifulSoup(html,"html5lib")
#找到所有文章页面
show_all = bf.find_all('div',class_ = 'show-all')
href = url + show_all[0].a.get('href')
#进入所有文章页面
req = requests.get(url = href)
html = req.text


# In[3]:


bf = BeautifulSoup(html,"html5lib")


# In[8]:


#找到要抓取文章页面
paperTitle = bf.find_all('a',class_ = 'title')

#paperNum为抓取的最新文章数，目前最大值为20，大于此数值时需要切换下一页
#paperNum = len(paperTitle) #抓取本页面所有文章（20篇）
paperNum = 5 #抓取前5篇文章
targets = []
for i in range(paperNum):
    targets.append(url + paperTitle[i].get('href'))
targets


# In[9]:

#抓取文章
papers = []
for i in range(paperNum):
    papers.append(getPaperInfo(targets[i]))


# In[22]:
#结果为list，每个list元素为一个dictionary，分别有title，abstract，keywords字段
print(papers)

