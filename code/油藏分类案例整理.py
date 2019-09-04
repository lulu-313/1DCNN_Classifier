# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:49:41 2018
9-28
@author: 李艳春
目的：油藏模型分类

方法：采用试井压力导数曲线作为训练数据，通过keras构造分类模型


"""
# 导入基本库
import cx_Oracle
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy.random import seed 
seed(1) 
#from tensorflow import set_random_seed 
#set_random_seed(2)

# LabelEncoder 用来编码输出标签
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# StratifiedShuffleSplit可以用来把数据集洗牌，并拆分成训练集和验证集
from sklearn.model_selection import StratifiedShuffleSplit

# 我们用的Keras支持模型创建
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout,AveragePooling1D,MaxPooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LambdaCallback
from keras.callbacks import EarlyStopping


#%%
# Connect to the database 连接数据库读取类型数据
#dsn=cx_Oracle.makedsn("192.168.1.129", 1521, "bzsjjspt")  
#conn=cx_Oracle.connect("bztest","bztest",dsn)  
#cursor=conn.cursor()  

#压力导数曲线数据
rr =  pd.read_csv(r'D:\python\workspace\welltest\classifier.csv', header=0)

point_df1 = pd.DataFrame(rr["NEW_ID"])
point_df2 = pd.DataFrame(rr["LOG_DP"].str.split(',',expand=True))
point = pd.concat([point_df1, point_df2.iloc[:, 0:93]], axis = 1)

#类型
species = pd.DataFrame(rr[["NEW_ID","SPECIES"]])

#%%
# 用LabelEncoder为油藏的种类标签编码
label_encoder = LabelEncoder().fit(species.SPECIES)
# labels对象是训练集上的标签列表
labels = label_encoder.transform(species.SPECIES)
# classes记录油藏类型
classes = list(label_encoder.classes_)
# 此处把训练集不必要的列删除
train = point.drop(['NEW_ID'], axis=1)

# 标准化数据集
scaler = StandardScaler().fit(train.values)
scaled_train = scaler.transform(train.values)

# 把数据集拆分成训练集和测试集，测试集占10%
sss = StratifiedShuffleSplit(test_size=0.1, random_state=23)
for train_index, valid_index in sss.split(scaled_train, labels):
    X_train, X_valid = scaled_train[train_index], scaled_train[valid_index]
    y_train, y_valid = labels[train_index], labels[valid_index]

# 每个输入通道的大小是31位，一共3个通道
nb_features = 31 
nb_class = len(classes)

#  把输入数据集reshape成keras喜欢的格式：（样本数，通道大小，通道数）
X_train_r = np.zeros((len(X_train), nb_features, 3))

# 这里的做法是先把所有元素初始化成0之后，再把刚才的数据集中的数据赋值过来
X_train_r[:, :, 0] = X_train[:, :nb_features]
X_train_r[:, :, 1] = X_train[:, nb_features:nb_features*2]
X_train_r[:, :, 2] = X_train[:, nb_features*2:]

# 验证集也要reshape一下
X_valid_r = np.zeros((len(X_valid), nb_features, 3))
X_valid_r[:, :, 0] = X_valid[:, :nb_features]
X_valid_r[:, :, 1] = X_valid[:, nb_features:nb_features*2]
X_valid_r[:, :, 2] = X_valid[:, nb_features*2:]

# 将类别由整型标签转为onehot
# 使用one hot编码器对类别进行“二进制化”操作
y_train_labled=y_train
y_valid_labled=y_valid
y_train = np_utils.to_categorical(y_train)
y_valid = np_utils.to_categorical(y_valid, nb_class)# 不写默认生成数组长度相同的矩阵

#%%
# 运用Keras对一维卷积实现
model = Sequential()

# 一维卷积层用了512个卷积核，输入是31*3的格式
# 构造模型
# 卷积层1，512个过滤器，过滤器长度为5，输入长度为nb_features，纬度为3
with tf.name_scope('Covn-layer-1'):
    model.add(Convolution1D(nb_filter=512, filter_length=5, input_shape=(nb_features, 3)))
# 激活层1 relu函数
with tf.name_scope('Activation-layer-1'):
    model.add(Activation('relu'))
# 池化层1 采用平均池化 核大小为2 pad方式为valid
with tf.name_scope('AveragePooling-layer'):
    model.add(AveragePooling1D(pool_size=2, strides=None, padding='valid'))
# 卷积层2，128个过滤器，过滤器长度为5，输入长度为nb_features，纬度为3
with tf.name_scope('Covn-layer-2'):
    model.add(Convolution1D(nb_filter=128, filter_length=5, input_shape=(nb_features, 3)))
# 激活层2 relu函数
with tf.name_scope('Activation-layer-2'):
    model.add(Activation('relu'))
# 池化层1 采用最大池化 核大小为2 pad方式为valid
with tf.name_scope('MaxPooling-layer'):
    model.add(MaxPooling1D(pool_size=2, strides=None, padding='valid'))
# 平铺层，调整维度适应全链接层
with tf.name_scope('Dense-layer-1'):
    model.add(Flatten())
# 进行数据局部失活，防止过拟合
    model.add(Dropout(0.4))
# 全连接层
    model.add(Dense(2048, activation='relu'))
# 全连接层
with tf.name_scope('Dense-layer-2'):
    model.add(Dense(1024, activation='relu'))
with tf.name_scope('output'):
    model.add(Dense(nb_class))

# softmax经常用来做多类分类问题
with tf.name_scope('softmax'):
    model.add(Activation('softmax'))
    
# 编译模型
# 采用随机梯度下降优化函数
sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

#%% 自定义的类来保存日志记录
# 将on_train_run作为一个类回调打印模型
class Model_Summary(keras.callbacks.Callback):
    def on_train_begin(self,logs=None):
        print('On_train_begin')
        model.summary()
        print(keras.utils.layer_utils.print_summary(self.model))

# 调用tensorboard生成模型graph
class Model_Graph(keras.callbacks.TensorBoard):
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', 
                                         histogram_freq=0, 
                                         write_graph=True, 
                                         write_images=True)

# 输出损失值与正确率
show_loss_callback = LambdaCallback(
        on_epoch_end = lambda epoch,logs:
            print(epoch,
                  'loss:',logs['loss'],
                  'acc:',logs['acc']))# type(epoch),type(logs['loss']),type(logs['acc'])

# 回调记录损失值和变化率
class Model_History(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])
        self.acc.append(logs['acc'])
        self.val_acc.append(logs['val_acc'])
    
    # 清空画布
    plt.close('all')
    
    
    # 显示损失值变化情况
    def show_losss(self):
        plt.figure(1)
        plt.plot(np.arange(len(self.losses)),self.losses,label='losses')
        plt.plot(np.arange(len(self.val_losses)),self.val_losses,label='val_losses')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.title('model loss')
       
        
    # 显示正确率变化情况
    def show_accuracy(self):
        plt.figure(2)
        plt.plot(np.arange(len(self.acc)),self.acc,label='acc')
        plt.plot(np.arange(len(self.val_acc)),self.val_acc,label='val_acc')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.title('model accuracy')
        
#%%
#用于实现callback 保存日志记录   
history = Model_History()

#%%

# 模型训练的轮数
nb_epoch =50
patience = 4
# 训练模型
model_history = model.fit(X_train_r, y_train, 
                    batch_size=16,
                    epochs=nb_epoch,
                    verbose=1,
                    validation_data=(X_valid_r, y_valid),
                    # 是否在训练过程中随机打乱输入样本的顺序
                    shuffle=True,
                    callbacks = [Model_Summary(),
                                 Model_Graph(),
                                 EarlyStopping(patience=patience,mode='min',verbose=0),
                                 show_loss_callback,
                                 history])
score = model.evaluate(X_valid_r, y_valid, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
history.show_losss()
history.show_accuracy()
#
#model.save('D:\python\workspace\well\classical_model_epoch'+str(nb_epoch)+'.h5')   
#model.save_weights('D:\python\workspace\well\classical_model_weight_epoch'+str(nb_epoch)+'.h5')

#%%
# 进行预测
pred_y = model.predict(X_train_r)
#pre_result=np.print(pred_y)
result=list(pred_y.nonzero())[1]
