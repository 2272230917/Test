#coding=utf-8
import csv
import datetime
from io import StringIO
import tensorflow as tf
import web
import pymysql
import hashlib
import matplotlib.pyplot as plt
import time
import math
from sklearn import linear_model,metrics
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
# from tensorflow import keras
from tensorflow import keras
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

urls = (
    '/tnb.html', 'tnb',
    '/pred.html', 'pred',
    '/index.html', 'index',
    '/','login',
    '/register','register',
    '/dashboard.html','dashboard',
    '/screen.html','screen',
    '/log.html','log',
    '/sug.html','sug'
)
class log:
    def GET(self):
        dataOP = sqlSelect("select * from operationlog")
        dataTR = sqlSelect("select * from trainlog")
        return render.log(dataOP,dataTR)
    def POST(self):
        dataOP = sqlSelect("select * from operationlog")
        dataTR = sqlSelect("select * from trainlog")
        return render.log(dataOP,dataTR)
class register:
    def GET(self):
        return render.register("")
    def POST(self):
        data = web.input(nickname="",username="",password="", confirmPassword="")
        # 检查密码和确认密码是否一致
        if data.password != data.confirmPassword:
            return render.register(notice="密码和确认密码不匹配")
        print("select name from user where name='%s'" % data.nickname)
        selectNickName = sqlSelect("select name from user where name='%s'" % data.nickname)
        print("select username from user where username='%s'" % data.nickname)
        if len(selectNickName)!=0:
            return render.register(notice="昵称已存在!")
        selectUserName = sqlSelect("select username from user where username='%s'" % data.username)
        if len(selectUserName) != 0:
            return render.register(notice="用户名已存在!")
        m = hashlib.md5()
        m.update(data.password.encode("utf-8"))
        password = m.hexdigest()
        print("insert into user values ('%s','%s','%s','%s')"%((data.nickname,'users',data.username,password)))
        sqlWrite("insert into user values ('%s','%s','%s','%s')"%((data.nickname,'users',data.username,password)))
        raise web.seeother('/','注册成功')
class login:
    def GET(self):
        return render.login("")
    def POST(self):
        webData = web.input()
        print( webData.get("username"))
        username = webData.get("username")
        password = webData.get("password")
        #计算密码的md5值
        m = hashlib.md5()
        m.update(password.encode("utf-8"))
        password = m.hexdigest()
        sql = "select name,role from user where username ='%s' and password='%s'"%(username,password)
        sqlData = sqlSelect(sql)
        print(sqlData)
        if len(sqlData)==0:
            return render.login("密码错误")
        else:
            session.username = username
            session.name = sqlData[0][0]
            session.role = sqlData[0][1]

            raise web.seeother('/index.html')
class dashboard:
    def GET(sef):
        webData = web.input()
        pageSize = 10
        curPage = int(webData.get("page","1"))

        sql = "select count(*) from patients"
        totalCnt = sqlSelect(sql)[0][0]
        pageCnt = math.ceil(totalCnt/pageSize)
        sql="select id,PREGNANCIES,GLUCOSE,BLOODPRESSURE,SKINTHICKNESS,INSULIN,BMI,DIABETESPEDIGREEFUNCTION,AGE,OUTCOME from patients LIMIT %s OFFSET %s"%(pageSize, (curPage-1)*pageSize )
        data=sqlSelect(sql)

        return render.dashboard(data,curPage,pageCnt,["","","","","","",""])

    def POST(self):
        webData=web.input(myFile={})#name=myFile
        preduct=["","","","","","",""]
        modelInfo = ["", "", "", "", "","", ""]
        if "addBtn" in webData:
            #获取html中name=addPREGNANCIES
            PREGNANCIES=float(webData.get("addPREGNANCIES"))
            GLUCOSE=float(webData.get("addGLUCOSE"))
            BLOODPRESSURE=float(webData.get("addBLOODPRESSURE"))
            SKINTHICKNESS=float(webData.get("addSKINTHICKNESS"))
            INSULIN=float(webData.get("addINSULIN"))
            BMI=float(webData.get("addBMI"))
            DIABETESPEDIGREEFUNCTION=float(webData.get("addDIABETESPEDIGREEFUNCTION"))
            AGE=float(webData.get("addAGE"))
            OUTCOME=float(webData.get("addOUTCOME"))

            sql="INSERT INTO patients (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome)values (%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)"%(PREGNANCIES,GLUCOSE,BLOODPRESSURE,SKINTHICKNESS,INSULIN,BMI,DIABETESPEDIGREEFUNCTION,AGE,OUTCOME)
            print(sql)
            roleSql = "select * from user where username=%s"%session.username
            role = sqlSelect(roleSql)[0][1]
            current_datetime = datetime.datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            sqlLog= "insert into operationlog (username, role, date, operation) values (\'%s\',\'%s\',\'%s\',\'%s\')"%(session.name,role,formatted_datetime,"新增数据")
            sqlWrite(sqlLog)
            sqlWrite(sql)

        elif "updateBtn" in webData:
            upid = float(webData.get("updateId"))
            upPREGNANCIES = float(webData.get("updatePREGNANCIES"))
            upGLUCOSE = float(webData.get("updateGLUCOSE"))
            upBLOODPRESSURE = float(webData.get("updateBLOODPRESSURE"))
            upSKINTHICKNESS = float(webData.get("updateSKINTHICKNESS"))
            upINSULIN = float(webData.get("updateINSULIN"))
            upBMI = float(webData.get("updateBMI"))
            upDIABETESPEDIGREEFUNCTION = float(webData.get("updateDIABETESPEDIGREEFUNCTION"))
            upAGE = float(webData.get("updateAGE"))
            upOUTCOME = float(webData.get("updateOUTCOME"))
            sql="UPDATE patients SET Pregnancies=%.3f, Glucose=%.3f, BloodPressure=%.3f, SkinThickness=%.3f, Insulin=%.3f, BMI=%.3f, DiabetesPedigreeFunction=%.3f, Age=%.3f, Outcome=%.3f WHERE ID=%d"%(upPREGNANCIES,upGLUCOSE,upBLOODPRESSURE,upSKINTHICKNESS,upINSULIN,upBMI,upDIABETESPEDIGREEFUNCTION,upAGE,upOUTCOME,upid)
            sqlWrite(sql)
            roleSql = "select * from user where username=%s"%session.username
            role = sqlSelect(roleSql)[0][1]
            current_datetime = datetime.datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            sqlLog= "insert into operationlog (username, role, date, operation) values (\'%s\',\'%s\',\'%s\',\'%s\')"%(session.name,role,formatted_datetime,"修改数据")
            sqlWrite(sqlLog)
        elif "deleteBtn" in webData:
            id=float(webData.get("deleteId"))
            sql="delete from  patients where id=%.4f"%(id)
            sqlWrite(sql)
            roleSql = "select * from user where username=%s" % session.username
            role = sqlSelect(roleSql)[0][1]
            current_datetime = datetime.datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            sqlLog = "insert into operationlog (username, role, date, operation) values (\'%s\',\'%s\',\'%s\',\'%s\')" % (session.name, role, formatted_datetime, "删除数据")
            sqlWrite(sqlLog)
        elif "fileBtn" in webData:
            #fobj=open()
            fobj=StringIO(str(webData["myFile"].file.read(),encoding="utf-8"))
            fobj.readline()
            reader=csv.reader(fobj)
            count = 0
            for t in reader:
                print(t)
                sql="INSERT INTO patients (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome)values (%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)"%(float(t[0]),float(t[1]),float(t[2]),float(t[3]),float(t[4]),float(t[5]),float(t[6]),float(t[7]),float(t[8]))
                sqlWrite(sql)
                count = count+1
            if count>0:
                roleSql = "select * from user where username=%s" % session.username
                role = sqlSelect(roleSql)[0][1]
                current_datetime = datetime.datetime.now()
                formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                sqlLog = "insert into operationlog (username, role, date, operation) values (\'%s\',\'%s\',\'%s\',\'%s\')" % (session.name, role, formatted_datetime, "新增数据")
                sqlWrite(sqlLog)
        elif "train" in webData:
            modelName = webData.get("model","lr")
            sql="select PREGNANCIES,GLUCOSE,BLOODPRESSURE,SKINTHICKNESS,INSULIN,BMI,DIABETESPEDIGREEFUNCTION,AGE,OUTCOME from patients "
            sqlData = sqlSelect(sql)

            dataX = np.array(sqlData)[:,0:-1]
            dataY = np.array(sqlData)[:,-1]

            # 划分数据集
            train_x, test_x, train_y, test_y = train_test_split(dataX, dataY, test_size=0.2)
            NeuralNetworks = keras.Sequential([
                keras.layers.Dense(128, activation='relu', input_shape=(train_x.shape[1],)),
                keras.layers.Dropout(0.5),  # 添加Dropout层以减少过拟合
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(1, activation='sigmoid')  # 输出层，使用sigmoid激活函数进行二分类
            ])
            NeuralNetworks.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            modelDict = {
                "lr": RandomForestClassifier(max_depth=None, min_samples_split=2, n_estimators=50),
                "svr": DecisionTreeClassifier(max_depth=30, min_samples_split=2),
                "lasso": NeuralNetworks,
                "ridge": LogisticRegression(C=0.1, penalty='l2'),
            }

            model = modelDict[modelName]

            start_time = time.time() # 获取当前时间，作为训练开始时间

            model.fit(train_x, train_y) # 训练模型
            end_time = time.time() # 获取当前时间，作为训练结束时间
            #保存模型
            if modelName == 'lasso':
                # joblib.dump(model, "./static/%s.model" % modelName, protocol=4)
                print(1)
            else:
                joblib.dump(model, "./static/%s.model"%modelName, protocol=4)
            roleSql = "select * from user where username=%s" % session.username
            role = sqlSelect(roleSql)[0][1]
            current_datetime = datetime.datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

            timeCost = end_time-start_time

            modelDict = {
                "lr": RandomForestClassifier(max_depth=None, min_samples_split=2, n_estimators=50),
                "svr": DecisionTreeClassifier(max_depth=30, min_samples_split=2),
                "lasso": NeuralNetworks,
                "ridge": LogisticRegression(C=0.1, penalty='l2'),
            }
            if modelName=="lr":
                sqlTrainLog = "INSERT INTO trainlog(username, role, date, model,timecost) VALUES ('%s', '%s', '%s', '%s','%s');"%(session.name,role,formatted_datetime,"RandomForestClassifier",timeCost)
                sqlWrite(sqlTrainLog)
            elif modelName=="svr":
                sqlTrainLog = "INSERT INTO trainlog(username, role, date, model,timecost) VALUES ('%s', '%s', '%s', '%s','%s');"%(session.name,role,formatted_datetime,"DecisionTreeClassifier",timeCost)
                sqlWrite(sqlTrainLog)
            elif modelName == "lasso":
                sqlTrainLog = "INSERT INTO trainlog(username, role, date, model,timecost) VALUES ('%s', '%s', '%s', '%s','%s');"%(session.name,role,formatted_datetime,"NeuralNetworks",timeCost)
                sqlWrite(sqlTrainLog)
            elif modelName == "ridge":
                sqlTrainLog = "INSERT INTO trainlog(username, role, date, model,timecost) VALUES ('%s', '%s', '%s', '%s','%s');"%(session.name,role,formatted_datetime,"LogisticRegression",timeCost)
                sqlWrite(sqlTrainLog)
            # 预测并评价模型
            pred = model.predict(test_x)
            mae = metrics.mean_absolute_error(test_y, pred)
            mse = metrics.mean_squared_error(test_y, pred)
            rmse = metrics.mean_squared_error(test_y, pred, squared=False)
            r2 = metrics.r2_score(test_y, pred)
            lens=len(dataX)
            train_time=end_time - start_time
            if modelName=="lasso":
                modelInfo = [mae,mse,rmse,r2,lens,train_time,"NeuralNetworks"]
            else:
                modelInfo = [mae, mse, rmse, r2, lens, train_time, model]
        webData = web.input()
        pageSize = 10
        curPage = int(webData.get("page","1"))

        sql = "select count(*) from patients"
        totalCnt = sqlSelect(sql)[0][0]
        pageCnt = math.ceil(totalCnt/pageSize)
        sql="select id,PREGNANCIES,GLUCOSE,BLOODPRESSURE,SKINTHICKNESS,INSULIN,BMI,DIABETESPEDIGREEFUNCTION,AGE,OUTCOME from patients LIMIT %s OFFSET %s"%(pageSize, (curPage-1)*pageSize )
        data=sqlSelect(sql)

        return render.dashboard(data,curPage,pageCnt,modelInfo)

        
class index:
    def GET(self):
        
        sql = "select Outcome,count(*) from patients group by Outcome order by CONVERT(Outcome,SIGNED) "
        sqlData2 = sqlSelect(sql)
        x = [t[0] for t in sqlData2]
        y = [t[1] for t in sqlData2]
        plt.figure(figsize=(10,5))
        plt.bar(x,y)
        fname = "./static/graph/%s.png"%(int(time.time()))
        plt.savefig(fname)

        sql = "select * from patients limit 15"
        sqlData = sqlSelect(sql)
        return render.index(sqlData,fname,x,y)

class tnb:
    def GET(self):
        webData = web.input() #表单数据，dict形式
        id = webData.get("id","154")
        sql = "select * from patients where ID='%s'"%(id)
        sqlData = sqlSelect(sql)
        model = joblib.load("./RandomForest.pkl")
        features = [float(value) for value in sqlData[0][0:]]
        print(features)
        if features[9] == 1:
            features[9] = "是"
        else:
            features[9] = "否"
        features = [features]
        features[0][0] = int(features[0][0])
        agePred = model.predict([features[0][1:9]])
        if agePred[0] >= 0.5 :
            agePred="是"
        else:
            agePred="否"
        agePred = [agePred]
        print(agePred)
        return render.tnb(features,agePred)

    def POST(self):
        webData = web.input() #表单数据，dict形式
        # name = webData["inputName"]
        id = webData.get("inputId","1")
        sql = "select * from patients where ID='%s'"%(id)
        sqlData = sqlSelect(sql)
        model = joblib.load("./RandomForest.pkl")
        features = [float(value) for value in sqlData[0][0:]]
        print(features)
        if features[9] == 1:
            features[9] = "是"
        else:
            features[9] = "否"
        features = [features]
        features[0][0] = int(features[0][0])
        agePred = model.predict([features[0][1:9]])
        print(agePred)
        if agePred[0] >= 0.5 :
            agePred="是"
        else:
            agePred="否"
        agePred = [agePred]
        return render.tnb(features, agePred)

class pred:
    def GET(self):
        webData = web.input()
        modelName=webData.get("model","lr")
        data = ['1','0.455','0.365','0.095','0.514','0.2245','0.101','0.15']
        return render.pred(data,modelName,"???")

    def POST(self):
        webData = web.input()
        modelName = webData.get("model")
        print(modelName)
        
        data = [
            float(webData.get("f1")),
            float(webData.get("f2")),
            float(webData.get("f3")),
            float(webData.get("f4")),
            float(webData.get("f5")),
            float(webData.get("f6")),
            float(webData.get("f7")),
            float(webData.get("f8")),
        ]

        if modelName == "lasso":
            model = tf.keras.models.load_model('./static/%s.h5'%modelName)
        else:
            model = joblib.load("./static/%s.pkl"%modelName)
        agePred = model.predict([data])[0]

        print(agePred)
        if agePred >= 0.5 :
            agePred="是"
        else:
            agePred="否"
        agePred = agePred
        return render.pred(data,modelName,agePred)
    
class screen:
    def GET(self):
        haveDiabetesSql = sqlSelect("SELECT Outcome, COUNT(*) AS count FROM patients GROUP BY Outcome ORDER BY Outcome;")
        haveDiabetes = {
            "radius": '40%',
            'activeRadius': '45%',
            'data': [
                {
                    'name': '未患病',
                    'value': haveDiabetesSql[0][1]
                },
                {
                    'name': '患病',
                    'value': haveDiabetesSql[1][1]
                }

            ],
            'digitalFlopStyle': {
                'fontSize': 20
            },
            'showOriginValue': 'true'
        }

        sqlDataBarOption = sqlSelect("""SELECT Outcome, COUNT(*) AS count FROM patients where Outcome=1 GROUP BY Outcome ORDER BY Outcome;""")
        ageALL = []
        ageCount = []
        for item in sqlDataBarOption:
            ageALL.append(str(item[0]))
            ageCount.append(item[1])


        barOption = {
            'title': {
                'text': '患者年龄分布'
            },
            'xAxis': {
                'name': '年龄',
                'data': ageALL
            },
            'yAxis': {
                'name': '数量',
                'data': 'value'
            },
            'series': [
                {
                    'data': ageCount,
                    'type': 'bar'
                }
            ]
        }

        sqlDataOption1Date = sqlSelect("""SELECT
            Pregnancies,
            COUNT(*) AS count
        FROM
            patients
        WHERE
            Outcome = 1
        GROUP BY
            Pregnancies
        ORDER BY
            Pregnancies;""")
        option1Data = []
        for item in sqlDataOption1Date:
            temp = {'name':str(item[0])+"次",'value':item[1]}
            option1Data.append(temp)
        option1={
            'title': {
                'text': '妊娠次数',
                'textStyle': {
                    'fontWeight': 'bold'
                },
                'itemStyle': {
                    'color': 'red!important'
                }

            },

            'series': [
                {
                    'type': 'pie',
                    'data': option1Data,
                    'insideLabel': {
                        'show': 'true'
                    }
                }
            ]
        }

        sqlData = sqlSelect("""
        SELECT 
        MAX(Pregnancies) AS max_Pregnancies, 
        MIN(Pregnancies) AS min_Pregnancies, 
        MAX(Glucose) AS max_Glucose, 
        MIN(Glucose) AS min_Glucose, 
        MAX(BloodPressure) AS max_BloodPressure, 
        MIN(BloodPressure) AS min_BloodPressure, 
        MAX(SkinThickness) AS max_SkinThickness, 
        MIN(SkinThickness) AS min_SkinThickness, 
        MAX(Insulin) AS max_Insulin, 
        MIN(Insulin) AS min_Insulin, 
        MAX(BMI) AS max_BMI, 
        MIN(BMI) AS min_BMI, 
        MAX(DiabetesPedigreeFunction) AS max_DiabetesPedigreeFunction, 
        MIN(DiabetesPedigreeFunction) AS min_DiabetesPedigreeFunction, 
        MAX(Outcome) AS max_Outcome, 
        MIN(Outcome) AS min_Outcome
        FROM patients
        WHERE Outcome=1;
        """)
        print(sqlData)
        config34Data = []
        new_column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                            'DiabetesPedigreeFunction', 'Outcome']
        count = 0
        for item in new_column_names:
            max = sqlData[0][count]
            count = count + 1
            min = sqlData[0][count]
            config34Data.append([item, min, max])
            count = count + 1
        config34 = {
            'header': ['指标', '最小值', '最大值'],
            'data':config34Data
        }


        sqlData = sqlSelect("""
        SELECT
            ROUND(AVG(Pregnancies)) AS avg_Pregnancies,
            ROUND(AVG(Glucose)) AS avg_Glucose,
            ROUND(AVG(BloodPressure)) AS avg_BloodPressure,
            ROUND(AVG(SkinThickness)) AS avg_SkinThickness,
            ROUND(AVG(Insulin)) AS avg_Insulin,
            ROUND(AVG(BMI)) AS avg_BMI,
            ROUND(AVG(DiabetesPedigreeFunction)) AS avg_DiabetesPedigreeFunction,
            ROUND(AVG(Outcome)) AS avg_Outcome
        FROM
            patients
        WHERE Outcome=1;
        """)
        config35 = []
        new_column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                            'DiabetesPedigreeFunction', 'Outcome']
        count = 0
        for item in new_column_names:
            avgNum = sqlData[0][count]
            count = count + 1
            config35.append({'name': item, 'value': int(avgNum)})

        config35 = {
            'data': config35
        }

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
        tableConfig={
            'header': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Outcome', 'Outcome'],
            'data':allData,
            'index': 'true',
            'columnWidth': [],
            'rowNum': 10,
            'align': ['center']
        }
        return render.screen(haveDiabetes,barOption,option1,config34,config35,tableConfig)

class sug:
    def GET(self):
        return render.sug()
    def POST(self):
        return render.sug()



render = web.template.render('templates/')
web.config.debug = False

app = web.application(urls, globals())
# root = tempfile.mkdtemp()
# store = web.session.DiskStore(root)
# session = web.session.Session(app, store)
#2272230917 hanhan7410
# activate SparkStudent
# python index.py
session = web.session.Session(app,web.session.DiskStore("sessions"),initializer={"username":'root','role':None})
if __name__ == "__main__":
    app.run()
