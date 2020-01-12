
import reader
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Fishnet import FishNets
from alexnet_1 import alexnet
# import matplotlib.pyplot as plt
num_epoch = 10
num_classify = 6
learning_rate = 0.001
save_model="./tmp/train_model.ckpt"
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)
    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def onehot(label):
    n_sample=len(label)
    # n_class=max(label)+1
    onehot_labels=np.zeros((n_sample,num_classify))
    onehot_labels[np.arange(n_sample),label]=1
    return onehot_labels
# with tf.name_scope("accuracy"):
with tf.Graph().as_default(),tf.device("/cpu:0"):
    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    y = tf.placeholder(tf.float32, shape=[None, 6])
    global_step = tf.get_variable('global_variable', initializer=tf.constant(0),trainable=False)
    tower_grads = []
    losses=[]
    network_planes = [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600]
    num_res_blks = [2, 2, 6, 2, 1, 1, 1, 1, 2, 2]
    num_trans_blks = [1, 1, 1, 1, 1, 4]
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    with tf.variable_scope(tf.get_variable_scope()):
        # with tf.device("/gpu:0"):
            # fc3 = alexnet(x, num_classify)
        with tf.name_scope("name",) as scope:
            # value=alexnet(x,6)
            mode=FishNets(num_classify,network_planes,num_res_blks,num_trans_blks)
            value=mode(x,training= False)
            a=tf.arg_max(value,1)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=value, labels=y))
    #         # train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    #         tf.get_variable_scope().reuse_variables()
    #         summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
    #         gradient = optimizer.compute_gradients(loss)
    #         tower_grads.append(gradient)
    # grads = average_gradients(tower_grads)
    # for grad, var in grads:
    #     if grad is not None:
    #         summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    #
    # train_op = optimizer.apply_gradients(grads,global_step=global_step)
    accuracy = tf.equal((tf.argmax(value, 1),),
                        tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    saver = tf.train.Saver()
    # summary_op = tf.summary.merge(summaries)
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False))

    print("begin")
    TRAIN_FILE = "./Classify.tfrecords"
    read=reader.Reader(TRAIN_FILE,batch_size=24)
    image_dataset,label_dataset=read.feed()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())


    saver.restore(sess,"./tmp/train_model.ckpt")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    # summary_writer = tf.summary.FileWriter("./log", sess.graph)
    for step in range(12000):# 83个epoch
            batch_label,batch_image=sess.run([label_dataset,image_dataset])

            if step % 1==0:
                print(batch_label)
            batch_label = onehot(batch_label)

            values, accu, los = sess.run([ a, accuracy, loss], feed_dict={
                x: batch_image, y: batch_label})

            # values,optimize, accu, los = sess.run([a,train_op, accuracy, loss], feed_dict={
            #
            #                        x: batch_image, y: batch_label})

            # summary_writer.add_summary(summary_str, step)
            if step %1==0:
                #summary_str=sess.run(summary_op,feed_dict={
                                   # x: batch_image, y: batch_label})
                # summary_writer.add_summary(summary_str, step)
                print(" %d     准确率为%f     损失为%f   " % (step,accu,los))
                print(values)

            if step % 100==0:
                saver.save(sess,save_model)
    coord.request_stop()
    coord.join(threads)
    plt.plot(losses)
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.savefig('./cnn-tf-AlexNet.png',dpi=200)
