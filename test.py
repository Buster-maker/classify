import os
from PIL import Image
import tensorflow as tf
from Fishnet import FishNets
import numpy as np
import json
def onehot(label):
  n_sample = len(label)
  # n_class=max(label)+1
  onehot_labels = np.zeros((n_sample, 6))
  onehot_labels[np.arange(n_sample), label] = 1
  return onehot_labels
def read(file_list):
    # 构建文件名队列
    x = tf.placeholder(tf.float32, [None, 224,224,3])
    file_queue=tf.train.string_input_producer(file_list)

    # 读取与解码
    reader=tf.WholeFileReader()
    _,value=reader.read(file_queue)
    image_a=tf.image.decode_jpeg(value,channels=3)
    image=tf.image.resize_images(image_a,[224,224])
    image=tf.cast(image,tf.float32)
    image=tf.reshape(image,shape=[224,224,3])
    # 批处理
    inputs=tf.train.batch([image],batch_size=22,num_threads=1,capacity=1)
    network_planes = [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600]
    num_res_blks = [2, 2, 6, 2, 1, 1, 1, 1, 2, 2]
    num_trans_blks = [1, 1, 1, 1, 1, 4]
    mode = FishNets(6, network_planes, num_res_blks, num_trans_blks)
    value = mode(x, training=True)
    va=tf.argmax(value,1)

    # saver = tf.train.import_meta_graph("./tmp/train_model.ckpt")
    saver=tf.train.Saver()
    with tf.Session() as sess:
        #model = tf.train.latest_checkpoint("./tmp")
        #print(model)
        # saver.recover_last_checkpoints("./tmp/train_model.ckpt")
        saver.restore(sess,save_path="./tmp/train_model.ckpt")
        cood=tf.train.Coordinator()
        thread=tf.train.start_queue_runners(sess=sess,coord=cood)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        inputs=sess.run(inputs)
        prediction,values=sess.run([va,value],feed_dict={x:inputs})
        for i in range(len(file_list)):
            print(prediction[i])

        # result=[]
        # for i in range(len(file_list)):  # file_list图片地址
        #     disease_dict={}
        #     pic_file=file_list[i]
        #     pic_file=pic_file[8:]
        #     disease_dict["image_id"] = pic_file
        #     disease_dict["disease_class"]=int(prediction[i])+1
        #     result.append(disease_dict)
        # with open ("./danyi.json",'w') as f:
        #     f.write(json.dumps(result))
        # print("done")
        cood.request_stop()
        cood.join(thread)
filename=os.listdir("./image")
file_list=[os.path.join("./image/",file) for file in filename]
print(file_list)
a=read(file_list)

















# def per_calss(imagefile):
   #  image=Image.open(imagefile)
    # image=image.resize([227,227])
    # image_array=np.array(image)
    # image=tf.cast(image_array,tf.float32)
    # image=tf.image.per_image_standardization(image)
    # image=tf.reshape(image,shape=[1,227,227,3])
    # saver=tf.train.Saver()
    # with tf.Session() as sess:
    #     save_model=tf.train.latest_checkpoint("./tmp")
    #     saver.restore(sess,save_model)
    #     image=sess.run(image)
    #     prediction=sess.run(fc3,feed_dict={x:image})
    #     max_index=np.argmax(prediction)
    #     print(max_index)

# filename=os.listdir("./IDADP-PRCV2019-training/1")
# print(filename)
# file_list=[os.path.join("./dog/",file) for file in filename]
# a=per_calss(file_list)



















# inputs=tf.nn.batch_normalization(inputs)
#     inputs_shape = inputs.get_shape().as_list()
#     batchsize, height, width, C = inputs_shape[0], inputs_shape[1], inputs_shape[2], inputs_shape[3]
#     filter = tf.Variable(tf.truncated_normal([1, 1, C, 1], dtype=tf.float32, stddev=0.1), name='weights')
#     filter1 = tf.Variable(tf.truncated_normal([1, 1, C, C], dtype=tf.float32, stddev=0.1), name='weights1')
#     query_conv = tf.nn.conv2d(inputs, filter, strides=[1, 1, 1, 1], padding='VALID')
#     print(query_conv)
#     key_conv = tf.nn.conv2d(inputs, filter, strides=[1, 1, 1, 1], padding='VALID')
#     print(key_conv)
#     value_conv = tf.nn.conv2d(inputs, filter1, strides=[1, 1, 1, 1], padding='VALID')
#     print(value_conv)
#     proj_query = tf.reshape(query_conv, [batchsize, width * height, -1])
#     print(proj_query)
#     proj_key = tf.transpose((tf.reshape(key_conv, [batchsize, width * height, -1])), perm=[0, 2, 1])
#     print(proj_key)
#     energy = tf.matmul(proj_query, proj_key)
#     print(energy)
#     attention = tf.nn.softmax(energy)
#     print(attention)
#     proj_value = tf.reshape(value_conv, [batchsize, width * height, -1])
#     print(proj_value)
#     out = tf.matmul(attention, proj_value)
#     print(out)
#     out = tf.reshape(out, [batchsize, height, width, C])
#     print(out)
#     # out = out + inputs