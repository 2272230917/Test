# #coding = utf-8
# import pandas as pd
# import warnings
# from sklearn import linear_model
# import joblib
#
# warnings.filterwarnings('ignore')
#
#
# df = pd.read_csv(r"./patients.txt",sep='\t',header=None)
# dataX = df.iloc[:,0:-1]
# dataY = df.iloc[:,-1]
#
# model = linear_model.LinearRegression()
# model.fit(dataX,dataY) #训练模型
#
# #保存模型
# joblib.dump(model, 'lr.model', protocol=2)
#
# pred = model.predict([['1','0.455','0.365','0.095','0.514','0.2245','0.101','0.15']])
# print(pred)
# import hashlib
# password = "hanhan7410"
# m = hashlib.md5()
# m.update(password.encode("utf-8"))
# password = m.hexdigest()
# print(password)
import datetime

import pymysql


def sqlSelect(sql):
    conn=pymysql.connect(host='localhost',port=3306,user='root',passwd='hanhan7410',db='web')
    cur = conn.cursor()
    cur.execute(sql)
    sqlData=cur.fetchall()
    cur.close()
    conn.close()
    return sqlData
def sqlWrite(sql):
    conn=pymysql.connect(host='localhost',port=3306,user='root',passwd='hanhan7410',db='web')
    cur = conn.cursor()
    cur.execute(sql)
    cur.close()
    conn.commit()
    conn.close()
    return


# print(sqlSelect("select * from user where username = '2272230917'")[0][1])
# current_datetime = datetime.datetime.now()
# formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
# print("insert into operationlog (username, role, date, operation) values ('htl','admin',\'%s\','新增数据')"%str(formatted_datetime))
# sqlWrite("insert into operationlog (username, role, date, operation) values ('htl','admin',\'%s\','新增数据')"%str(formatted_datetime))
# print(str(formatted_datetime))

sql = "select * from patients where ID='%s'" % (3)
sqlData = sqlSelect("""
SELECT
    *
FROM
    patients
limit 50;
""")
allData = []
for item in sqlData:
    allData.append(list(item[1:]))



