import tensorflow as tf
import numpy as np
class Reader():
    def __init__(self,tfrecord_file,image_size=224,min_queue_examples=100,
                 batch_size=1, num_threads=1,name=''):
        """
        :param tfrecord_file:        tfrecord文件路径
        :param image_size:           将图片resize为同样大小
        :param min_queue_examples:
        :param batch_size:            批处理图片的大小
        :param num_threads:            开启线程个数
        :param name:
        """
        self.tfrecord=tfrecord_file
        self.image_size=image_size,
        self.min_queue_example=min_queue_examples,
        self.batch_size=batch_size
        self.reader=tf.TFRecordReader()
        self.num_threads=num_threads
        self.name=name
    def feed(self):
        """
        :return:4Dtensor[batch_size,image_width, image_height,channels]
        """
        with tf.name_scope(self.name):
            # 读取文件名队列
            file_queue=tf.train.string_input_producer([self.tfrecord],num_epochs=10)
            # 将数据用TFRecordReader的方式读入value中
            _,value=self.reader.read(file_queue)
            # 将数据解码还原，放到张量里
            sample=[]
            features = tf.parse_single_example(value,features={
                "image":tf.FixedLenFeature([],tf.string),
                "label":tf.FixedLenFeature([],tf.int64)
            })
            # 获取图片和目标值
            image=features["image"]
            label=features["label"]

            # 将图片解码
            image=tf.image.decode_jpeg(image,channels=3)

            # 图片处理过程
            image=self.process(image)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
            # 批处理
            image_batch,lable_batch=tf.train.shuffle_batch([image,label],
                                                           batch_size=self.batch_size,
                                                           num_threads=self.num_threads,
                                                            capacity=1024,
                                                            min_after_dequeue=900
            )
            # image_batch, lable_batch = tf.train.batch([image,label],batch_size=32,num_threads=8,
            #                                  capacity=200)

            label_batch=tf.reshape(lable_batch,shape=[self.batch_size])
        return image_batch,label_batch

    def process(self,image):
        # 将图片处理到统一大小

        image=tf.image.resize_images(image,size=[224,224])
        # image=tf.image.random_flip_left_right(image)# 随机左右翻转
        # image=tf.image.random_flip_up_down(image)# 随机上下翻转
        # image=tf.image.rot90(image,np.random.randint(1,4))# 随机旋转90*n次
        # 将图片转化为float32类型
        image=tf.cast(image,tf.float32)
        # 将图片转化为三维的
        image=tf.reshape(image,shape=[224,224,3])
        # 将图片进行random剪切反转操作
        # 将图片归一化
        image=tf.image.per_image_standardization(image)
        return image
#
# def read():
#     TRAIN_FILE="./Classify.tfrecords"
#     read=Reader(TRAIN_FILE,batch_size=32,)
#     image_re,label_dataset=read.feed()
#     # 开启会话操作
#     sess=tf.Session()
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     try:
#         step = 0
#         while not coord.should_stop():
#             batch_image, batch_label=sess.run([image_re,label_dataset])
#             print(batch_image)
#
#
#             # print("image shape: {}".format(batch_images))
#             # print("=" * 10)
#             step += 1
#     except KeyboardInterrupt:
#         print('Interrupted')
#         coord.request_stop()
#     except Exception as e:
#         coord.request_stop(e)
#     finally:
#         # When done, ask the threads to stop.
#         # 回收线程
#         coord.request_stop()
#         coord.join(threads)
#     return label_dataset
# read()
