import os
from tensorflow.python import pywrap_tensorflow
'''
import tensorflow as tf
saver=tf.train.import_meta_graph('./tmp/train_model.ckpt.meta')
graph=tf.get_default_graph()
a_val=graph.get_tensor_by_name()
'''
'''
import os
import re
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

model_exp = "20180402-114759"
def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)  # 通过checkpoint文件找到模型文件名
    if ckpt and ckpt.model_checkpoint_path:
        # ckpt.model_checkpoint_path表示模型存储的位置，不需要提供模型的名字，它回去查看checkpoint文件
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


meta_file, ckpt_file = get_model_filenames(model_exp)

print('Metagraph file: %s' % meta_file)
print('Checkpoint file: %s' % ckpt_file)
reader = pywrap_tensorflow.NewCheckpointReader(os.path.join(model_exp, ckpt_file))
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    # print(reader.get_tensor(key))

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
    saver.restore(tf.get_default_session(),
                  os.path.join(model_exp, ckpt_file))
    print(tf.get_default_graph().get_tensor_by_name("Logits/weights:0"))
    '''



# 使用NewCheckpointReader来读取ckpt里的变量
from tensorflow.python import pywrap_tensorflow

checkpoint_path = './tmp/train_model.ckpt'
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  # tf.train.NewCheckpointReader
var_to_shape_map = reader.get_variable_to_shape_map()
var_to_shape_map=sorted(var_to_shape_map)
for i ,key in enumerate(var_to_shape_map):
    print("tensor_name: ", key)
    print(reader.get_tensor(key))

# 使用print_tensors_in_checkpoint_file打印ckpt里的内容
'''
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

print_tensors_in_checkpoint_file('./tmp/train_model.ckpt.data-00000-of-00001',  # ckpt文件名字
                                 None,  # 如果为None,则默认为ckpt里的所有变量
                                 True,  # bool 是否打印所有的tensor，这里打印出的是tensor的值，一般不推荐这里设置为False
                                True)  # bool 是否打印所有的tensor的name
# 上面的打印ckpt的内部使用的是pywrap_tensorflow.NewCheckpointReader所以，掌握NewCheckpointReader才是王道
'''