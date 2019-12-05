#coding=utf-8
import sys
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.saved_model import tag_constants


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

# read dataset into an array
# labels will be include
def read_data(data_dir):
    datas = []
    labels = []
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        image = Image.open(fpath)
        data = np.array(image) / 255.0
        label = int(fname.split("_")[0])
        datas.append(data)
        labels.append(label)

    datas = np.array(datas)
    labels = np.array(labels)

    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return fpaths, datas, labels


def cnn(data_dir, model_path, load_flag):
    # image size in data set
    rows = 32
    columns = 32

    fpaths, datas, labels = read_data(data_dir)

    # how many classes there are
    num_classes = len(set(labels))


    # Placeholder for input and labels
    datas_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_placeholder = tf.placeholder(tf.int32, [None])

    # DropOut, which is 0.25 for training, and is 0 for testing
    dropout_placeholdr = tf.placeholder(tf.float32)

    # convolutional layer, 20 kenerl size, stride is 5，relu
    conv0 = tf.layers.conv2d(datas_placeholder, 20, 5, activation=tf.nn.relu)
    # max-pooling layer，pooling window is 2x2，step is2x2
    pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])

    # convolution, 40 kernel size, stride is 4, activety fun is Relu
    conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
    # max-pooling layer，pooling window is 2x2，step is2x2
    pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])

    # 3D to 1D
    flatten = tf.layers.flatten(pool1)

    # full connection
    fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)

    # for overfit
    dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)

    logits = tf.layers.dense(dropout_fc, num_classes)

    predicted_labels = tf.arg_max(logits, 1)


    losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(labels_placeholder, num_classes),
        logits=logits
    )

    mean_loss = tf.reduce_mean(losses)

    # po
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)

    # for save and restore
    saver = tf.train.Saver()

    with tf.Session() as sess:

        if train:
            print("in training model")
            # initial args
            sess.run(tf.global_variables_initializer())
            if load_flag:
                saver.restore(sess, model_path)
                print("#####################")
                print("load {} model".format(model_path))
            # define input, Label and placeholder,dropout is 0.25 for training
            train_feed_dict = {
                datas_placeholder: datas,
                labels_placeholder: labels,
                dropout_placeholdr: 0.26
            }
            for step in range(150):
                _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=train_feed_dict)

                if step % 10 == 0:
                    print("step = {}\tmean loss = {}".format(step, mean_loss_val))
            save_path = saver.save(sess, model_path)
            print("training over {}".format(model_path))
        else:
            print("testing: ")
            # load model
            saver.restore(sess, model_path)
            print("#####################")
            print("load {} model".format(model_path))
            # label和名称的对照关系
            label_name_dict = {
    #            0: "not cat",
                3: "cat"
            }
            # 定义输入和Label以填充容器，测试时dropout为0
            test_feed_dict = {
                datas_placeholder: datas,
                labels_placeholder: labels,
                dropout_placeholdr: 0
            }
            predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
            # 真实label与模型预测label
            default = "not cat"
            for fpath, real_label, predicted_label in zip(fpaths, labels, predicted_labels_val):
                # 将label id转换为label名
                real_label_name = label_name_dict.get(real_label, default)
                predicted_label_name = label_name_dict.get(predicted_label, default)
#                print("{}\t{} => {}".format(fpath, real_label_name, predicted_label_name))
            result = [(a, b, c) for a,b,c in zip(fpaths, labels, predicted_labels_val) if c ==b]
            print("correct {} in {}".format(len(result), len(labels)))



if __name__ == '__main__':
    base_dir = "data"
    data_set = []
    # for training or testing
    if len(sys.argv) < 2:
        print("-t training_folder")
        print("-T testing_folder")
        print("folder must be in ./data")
        sys.exit()

    if sys.argv[1] == "-T":
        train = False
    else:
        train = True

    if "-d" in sys.argv:
        pos = sys.argv.index("-d")
        pos += 1
        while pos < len(sys.argv) and sys.argv[pos] in os.listdir(base_dir): 
            data_set.append( os.path.join(base_dir,sys.argv[pos]) )
            pos += 1
        if len(data_set) == 0:
            print("error: no folder after -d")
            sys.exit()
    else:
        print("-d is necessary")
        sys.exit()

    if "-lp" in sys.argv:
        model_path = sys.argv[sys.argv.index("-lp")+1]
    else:
        model_path = "./cnn_model"

    flag = False
    if "-ld" in sys.argv:
        flag = True

    for item in data_set:
        cnn(item, model_path, flag)

