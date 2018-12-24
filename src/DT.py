import csv
from sklearn.externals.six import StringIO
from sklearn import tree
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
#打开文件
file_open = open("D:\Java\Javacode\decision_Tree_20181224\AllElectronics.csv","rt")
#读取数据
read_data = csv.reader(file_open)

#读取表格第一行，一般都是标签那一行
header_data = next(read_data)
print(header_data)

#放特征值
featureList=[]
#放类别
labelList = []
for row in read_data:
    labelList.append(row[len(row)-1]) #把每一行的最后一个值添加进labelList中，也就是类别
    rowDict = {}
    for i in range(1,len(row)-1):
        rowDict[header_data[i]] = row[i]
    featureList.append(rowDict)
    
print(featureList)

vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print("dummyx:" + str(dummyX))
print(vec.get_feature_names())
print("labelList:" + str(labelList))

#将类别这一列中的数据转化成0、1类型的
lb=preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY:" + str(dummyY))

#上面的数据已经处理好了，接下来创建分类器
#选取哪个属性作为结点，就决定了用何种算法，这里使用两个信息熵的差来作为度量，
clf = tree.DecisionTreeClassifier(criterion='entropy')
#建模 
clf = clf.fit(dummyX,dummyY)
print("clf:" + str(clf))

#画图
with open("newData.dot",'w') as f:
    f = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)
    
oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))

predictedY = clf.predict(newRowX.reshape(1,-1))
print("predictedY: " + str(predictedY))

