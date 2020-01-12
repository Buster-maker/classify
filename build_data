import tensorflow as tf
import random
import os
from collections import namedtuple
from os import scandir
tf.flags.DEFINE_string("data_1","./IDADP-PRCV2019-training/1",
    "target为1的数据集")
tf.flags.DEFINE_string("data_2","./IDADP-PRCV2019-training/2",
    "target为2的数据集")
tf.flags.DEFINE_string("data_3","./IDADP-PRCV2019-training/3",
    "target为3的数据集")
tf.flags.DEFINE_string("data_4","./IDADP-PRCV2019-training/4",
    "target为4的数据集")
tf.flags.DEFINE_string("data_5","./IDADP-PRCV2019-training/5",
    "target为5的数据集")
tf.flags.DEFINE_string("data_6","./IDADP-PRCV2019-training/6",
    "target为6的数据集")
tf.flags.DEFINE_string("output_dir",'./Classify.tfrecords',
    "tfrecord保存数据集的地址")
tf.flags.DEFINE_string("pic_bise_dir", "path", "图片的基本地址")
FLAGS = tf.flags.FLAGS
ImageMetadata = namedtuple("ImageMetadata", ["pic_dir", "target"])
dataset = []


def data_process(path_dir, target):
    """
    :param data_dir:图片路径
    :param num: 图片对应的target
    :return:
    """
    # csandir 读取特定的目录文件
    for path_di in scandir(path_dir):
        # 判断图片是否以.JPG格式结尾,是否是常规文件
        if path_di.name.endswith('.JPG') and path_di.is_file():
            # 构建图片地址和目标值的对应关系
            a = ImageMetadata(path_di.path, target)
            dataset.append(a)


def convert_to_example(encoded_image, target):
    """
    :param encoded_image:read后的图片
    :param target: 目标值
    :return: 返回example协议
    """
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[encoded_image])), "label": tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[target]))}))
    return example


def write_in_tfrecord(dataset):
    """
    :param image:
    :return: 将图片写入文件
    """
    # 定义tfrecord的writer
    writer = tf.python_io.TFRecordWriter(FLAGS.output_dir)
    for image in dataset:
        # 读取图片
        with tf.gfile.FastGFile(image.pic_dir, "rb") as f:
            # 将图片变成原始编码形式
            encoded_image = f.read()
        # 将图片和target写入tfrecord
        example = convert_to_example(encoded_image, image.target)
        writer.write(example.SerializeToString())
    print("done")
    writer.close()
dir = [FLAGS.data_1,FLAGS.data_2,FLAGS.data_3,FLAGS.data_4,FLAGS.data_5,FLAGS.data_6]
path = dict([x, y] for x, y in enumerate(dir))
for i in range(6):
    path_dir = path[i]
    data_process(path_dir, i)
random.shuffle(dataset)
write_in_tfrecord(dataset)
