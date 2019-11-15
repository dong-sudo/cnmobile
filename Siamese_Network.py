import tensorflow as tf
import csv
import keras
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
import random
from keras.datasets import mnist
import random

#读取数据开始
def change_range(image):
  return 2*image-1

def load_and_preprocess_image(image_path):
    img1 = cv2.imread(image_path)
    img = img1[..., ::-1]
    img = img/255.0
    return change_range(img)


def load_csv(dir,o):
    p_list=[]
    l_list=[]
    with open(dir)as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            p_list.append(o+'\\'+row[0])
            l_list.append([int(i) for i in row[1:]] )
    return p_list,l_list



def set_database(pic_dir,path):
  data=[]
  random.seed(1)
  p,l=load_csv(pic_dir,path)
  for i in p:
      data.append(load_and_preprocess_image(i))
  return np.asarray(data),np.asarray(l)-1


#数据存入之前已经经过shuffle
#读取数据结束





#这个是提取feature的时候需要的参数

num_classes = 8
epochs = 20

#欧拉距离
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
#计算两者之间的距离

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

#将数据改造为triplelet的形式
def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

#构建简单孪生网络，也是最需要修改的地方，例如将网络本体化成efficienne
#t提取输出层之前的但是输出的dense最好保持128或者改成1024
def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = keras.applications.VGG16()(input)
    x = keras.layers.Dropout(0.1)(x)

    out = keras.layers.Dense(128, name='dense_layer',activation='relu')(x)
    return Model(input, out)

#构建自定的评估函数
def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))



# 输入cvs作为label和路径，将所有图片数据存在一个文件夹下，读入的图片为一通道的灰度图
x,y = set_database('C:\\test_data\\single_labels.csv','C:\\test_data\\resized_all_data')
x_test=x[:200]
y_test=y[:200]
x_train=x[201:]
y_train=y[201:]

input_shape = x_train.shape[1:]



digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
te_pairs, te_y = create_pairs(x_test, digit_indices)

# 定义神经网络
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

#该网络的输入与其他网络不同需要定义多个输入层之后拼接在一起
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
model.fit([tr_pairs[:, 0],tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

#以上是提取特征的网络

#建立分类器，提取出特征提取网络的参数，加上几个layer，我是用同一批数据进行的训练，
model = keras.Sequential()
for i in range(5):
    model.add(base_network.get_layer(index=i))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(8, name='dense_layer',activation='relu'))


model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.losses.sigmoid_cross_entropy,
              metrics=['accuracy'])

x,y = set_database('C:\\test_data\\single_for_keras.csv','C:\\test_data\\resized_all_data')
x_test=x[:200]
y_test=y[:200]
x_train=x[201:]
y_train=y[201:]
model.fit(x_train,y_train, epochs=10, steps_per_epoch=15)
