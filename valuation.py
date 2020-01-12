from PIL import Image
import numpy as np
import tensorflow as tf
def per_class(image_fiel):
    image=Image.open(image_fiel)
    image=image.resize(224,224)
    image_array=np.array(image)
    image= tf.cast(image_array,tf.float32)
    image=tf.image.per_image_standardization(image)
    image=tf.reshape(image,shape=[1,224,224,3])
    saver=tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,"")
        image=tf.reshape(image,shape=[1,224,224,3])
        iamge=sess.run(image)
        prediction=sess.run()


