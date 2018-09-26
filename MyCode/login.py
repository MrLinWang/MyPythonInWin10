# coding=utf-8
import urllib
import hashlib
import http.client
import http.cookiejar
import http.cookies
from tkinter import *

def ipv6Login(username,passwd):
    #post数据接收和处理的页面（我们要向这个页面发送我们构造的Post数据）
    posturl = 'https://lgn6.bjut.edu.cn/'  #从提交数据包中分析出，处理post请求的url
 
    #设置一个cookie处理器，它负责从服务器下载cookie到本地，并且在发送请求时带上本地的cookie
    cj = http.cookiejar.LWPCookieJar()
    cookie_support = urllib.request.HTTPCookieProcessor(cj)
    opener = urllib.request.build_opener(cookie_support, urllib.request.HTTPHandler)
    urllib.request.install_opener(opener)
    #打开登录主页面（他的目的是从页面下载cookie，这样我们在再送post数据时就有cookie了，否则发送不成功，当然有的网站也不对此做要求，那就省去这一步）
    h = urllib.request.urlopen('https://lgn6.bjut.edu.cn/')
 
    #构造header，一般header至少要包含一下两项。这两项是从抓到的包里分析得出的。
    headersIpv6 = {
        "Accept":'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        "Accept-Encoding":'gzip, deflate',
        "Accept-Language":'zh-CN',
        #"Cache-Control":'max-age=0',
        "Connection":'keep-alive',
        "DNT":'1',
        "Host":'lgn6.bjut.edu.cn',
        "Referer":'https://lgn6.bjut.edu.cn/F.htm',
        "Upgrade-Insecure-Requests":'1',
        "User-Agent":'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.79 Safari/537.36 Maxthon/5.2.4.3000'
    }
    #构造Post数据，也是从抓大的包里分析得出的。
 
    postData = {'DDDDD': username,
            'upass': passwd, #你的密码，密码可能是明文传输也可能是密文，如果是密文需要调用相应的加密算法加密,需查看js代码
            'v46s':'2',
            'v6ip':'',
            'f4serip':'172.30.201.10',
            '0MKKey':''                                                   #才知我就直接硬编码了
    }
 
    #需要给Post数据编码
    postData = urllib.parse.urlencode(postData).encode(encoding='UTF-8')
 
    #通过urllib.request提供的Request方法来向指定Url发送我们构造的数据，并完成登录过程
    request = urllib.request.Request(posturl, postData, headers)
    response = urllib.request.urlopen(request)

# # #测试是否成功登陆，这里是请求用户信息，如果成功登陆，那么cookie发到这个页面之后就会返回用户资料，否则提示错误，也知道自己登陆失败了
# # headers1 = {'User-Agent': ' Mozilla/5.0 (Windows NT 6.1; rv:32.0) Gecko/20100101 Firefox/32.0',
# #             'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
# #             'Accept-Language': ' zh-cn,zh;q=0.8,en-us;q=0.5,en;q=0.3',
# #             #  'Accept-Encoding': 'gzip, deflate',#大坑。加入后要手动对gzip解压后才会有可识别的内容
# #             'Referer': 'http://xxxx.edu.cn/ntms/userLogin.jsp?reason=logout',
# #             'Connection': 'keep-alive',
# #             'Content-Length': '0',
# #             'Content-Type': '    application/x-www-form-urlencoded; charset=UTF-8',
# #             'X-Requested-With': 'XMLHttpRequest'
# # }
# # request = urllib.request.Request('http://xxxx.edu.cn/action/getCurrentUserInfo.do', None, headers1)
# # response = urllib.request.urlopen(request)
# # text = response.read()
# # #打印回应的内容
# # print(str(text, encoding='utf-8', errors='strict'))

def ipv6Logout():
    urlIpv6Exit = 'https://lgn6.bjut.edu.cn/F.htm'

    headersIpv6Exit = {
        "Accept":'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        "Accept-Encoding":'gzip, deflate',
        "Accept-Language":'zh-CN',
        #"Cache-Control":'max-age=0',
        "Connection":'keep-alive',
        "DNT":'1',
        "Host":'lgn6.bjut.edu.cn',
        "Referer":'https://lgn6.bjut.edu.cn/',
        "Upgrade-Insecure-Requests":'1',
        "User-Agent":'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.79 Safari/537.36 Maxthon/5.2.4.3000'
    }

    #通过urllib.request提供的Request方法登出
    requestIpv6Exit = urllib.request.Request(urlIpv6Exit, headers=headersIpv6Exit)
    response = urllib.request.urlopen(requestIpv6Exit)


class MainWindow:
    def __init__(self):
        self.frame = Tk()

        self.label_username = Label(self.frame,text = "用户名:")
        self.label_passwd = Label(self.frame,text = "密码:")

        self.text_username = Text(self.frame,height = "1",width = 30)
        self.text_passwd = Text(self.frame,height = "1",width = 30)

        self.label_username.grid(row = 0,column = 0)
        self.label_passwd.grid(row = 1,column = 0)

        self.button_login = Button(self.frame,text = "登录",width = 10)
        self.button_cancel = Button(self.frame,text = "取消",width = 10)

        self.text_username.grid(row = 0,column = 1)
        self.text_passwd.grid(row = 1,column = 1)
        
        self.button_login.grid(row = 3,column = 0)
        self.button_cancel.grid(row = 3,column = 1)

        self.frame.mainloop()

    def login(self):
        name = self.label_username

frame = MainWindow()