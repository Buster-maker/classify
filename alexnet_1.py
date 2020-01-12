import tensorflow as tf
def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var
def batch_norm(inputs,is_training,is_conv_out=True,decay=0.999):
    scale=tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta=tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean=tf.Variable(tf.zeros([inputs.get_shape()[-1]]),trainable=False)
    pop_var=tf.Variable(tf.ones([inputs.get_shape()[-1]]),trainable=False)
    if is_training:
        if is_conv_out:
            batch_mean,batch_var=tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean,batch_var=tf.nn.moments(inputs,[0])
        train_mean=tf.assign(pop_mean,pop_mean*decay+batch_mean*(1-decay))
        train_var=tf.assign(pop_var,pop_var*decay+batch_var*(1-decay))
        with tf.control_dependencies([train_mean,train_var]):
            return tf.nn.batch_normalization(inputs,batch_mean,batch_var,beta,scale,0.001)
    else:
        return tf.nn.batch_normalization(inputs,pop_mean,pop_var,beta,scale,0.01)


def alexnet(x, num_classes):
    with tf.variable_scope('aaa'):
        weights = {
            'conv1': _variable_on_cpu('conv1', [11,11,3,96], tf.random_normal_initializer()),
            'conv2': _variable_on_cpu('conv2', [5,5,96,256], tf.random_normal_initializer()),
            'conv3': _variable_on_cpu('conv3', [3,3,256,384], tf.random_normal_initializer()),
            'conv4': _variable_on_cpu('conv4', [3, 3, 384, 384], tf.random_normal_initializer()),
            'conv5': _variable_on_cpu('conv5', [3, 3, 384, 256], tf.random_normal_initializer()),
            'fc1'  : _variable_on_cpu( 'fc1',  [6*6*256,4096], tf.random_normal_initializer()),
            'fc2'  : _variable_on_cpu('fc2',   [4096, 2048], tf.random_normal_initializer()),
            'fc3': _variable_on_cpu('fc3', [2048, 6], tf.random_normal_initializer()),
            # 'fc4': _variable_on_cpu('fc4', [256, 100], tf.random_normal_initializer()),
            #
            # 'fc5'  : _variable_on_cpu('fc5',   [100, 6], tf.random_normal_initializer())
        }

        biases = {
            'con1': _variable_on_cpu('con1', [96], tf.random_normal_initializer()),
            'con2': _variable_on_cpu('con2', [256], tf.random_normal_initializer()),
            'con3': _variable_on_cpu('con3', [384], tf.random_normal_initializer()),
            'con4': _variable_on_cpu('con4', [384], tf.random_normal_initializer()),
            'con5': _variable_on_cpu('con5', [256], tf.random_normal_initializer()),
            'f1': _variable_on_cpu('f1', [4096], tf.random_normal_initializer()),
            'f2': _variable_on_cpu('f2', [2048], tf.random_normal_initializer()),
            'f3': _variable_on_cpu('f3', [6], tf.random_normal_initializer()),
            # 'f4': _variable_on_cpu('f4', [100], tf.random_normal_initializer()),
            # 'f5': _variable_on_cpu('f5', [num_classes], tf.random_normal_initializer()),

        }
    # 卷积层1
    conv1=tf.nn.conv2d(x,weights['conv1'],strides=[1,4,4,1],padding='VALID')
    conv1=tf.nn.bias_add(conv1,biases['con1'])
    conv1=batch_norm(conv1,True)
    conv1=tf.nn.relu(conv1)
    pool1=tf.nn.avg_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
    norm1=tf.nn.lrn(pool1,5,bias=1.0,alpha=0.001/9.0,beta=0.75)
    # 卷积层2
    conv2=tf.nn.conv2d(norm1,weights['conv2'],strides=[1,1,1,1],padding='SAME')
    conv2=tf.nn.bias_add(conv2,biases['con2'])
    conv2=batch_norm(conv2,True)
    conv2=tf.nn.relu(conv2)
    pool2=tf.nn.avg_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
    # norm2=tf.nn.lrn(pool2,5,bias=1.0,alpha=0.001/9.0,beta=0.75)
    # 卷积层3
    conv3=tf.nn.conv2d(pool2,weights['conv3'],strides=[1,1,1,1],padding='SAME')
    conv3=tf.nn.bias_add(conv3,biases['con3'])
    conv3=batch_norm(conv3,True)
    conv3=tf.nn.relu(conv3)
    # 卷积层4
    conv4=tf.nn.conv2d(conv3,weights['conv4'],strides=[1,1,1,1],padding='SAME')
    conv4=tf.nn.bias_add(conv4,biases['con4'])
    conv4=batch_norm(conv4,True)
    conv4=tf.nn.relu(conv4)
    # 卷积层5
    conv5=tf.nn.conv2d(conv4,weights['conv5'],strides=[1,1,1,1],padding='SAME')
    conv5=tf.nn.bias_add(conv5,biases['con5'])
    conv5=tf.nn.relu(conv5)
    pool5=tf.nn.avg_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')

    reshape=tf.reshape(pool5,[-1,6*6*256])
    fc1=tf.add(tf.matmul(reshape,weights['fc1']),biases['f1'])
    fc1=batch_norm(fc1,False)
    fc1=tf.nn.relu(fc1)
    fc1=tf.nn.dropout(fc1,0.5)
    # 全连接
    fc2=tf.add(tf.matmul(fc1,weights['fc2']),biases['f2'])
    fc2=batch_norm(fc2,False)
    fc2=tf.nn.relu(fc2)
    fc2=tf.nn.dropout(fc2,0.5)
    fc3=tf.add(tf.matmul(fc2,weights['fc3']),biases['f3'])
    # fc3 = tf.nn.relu(fc3)
    # fc4 = tf.add(tf.matmul(fc3, weights['fc4']), biases['f4'])
    # fc4 = tf.nn.relu(fc4)
    # fc5 = tf.add(tf.matmul(fc4, weights['fc5']), biases['f5'])
    fc5=tf.nn.softmax(fc3)
    # 定义损失

    return fc5













