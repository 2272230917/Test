#coding=utf-8
import web
import pymysql
import hashlib
import tempfile

def sqlSelect(sql):
    conn=pymysql.connect(host='localhost',port=3306,user='root',passwd='123456',db='web')
    cur = conn.cursor()
    cur.execute(sql)
    sqlData=cur.fetchall()
    cur.close()
    conn.close()
    return sqlData
def sqlWrite(sql):
    conn=pymysql.connect(host='localhost',port=3306,user='root',passwd='123456',db='web')
    cur = conn.cursor()
    cur.execute(sql)
    cur.close()
    conn.commit()
    conn.close()
    return

urls = (
    '/login.html','login',
    '/', 'index',
)

class index:
    def GET(self):
        return render.xxx()
    def POST(self):
        return render.xxx()

render = web.template.render('templates/')
web.config.debug = False
app = web.application(urls, globals())
root = tempfile.mkdtemp()
store = web.session.DiskStore(root)
session = web.session.Session(app, store)
if __name__ == "__main__":
    app.run()
