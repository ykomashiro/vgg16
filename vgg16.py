import numpy as np
import tensorflow as tf
vgg16_npy_path = r'F:\Script\Data\vgg16.npy'


class VGG16:
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.prob = None
        self._out = None

    def _built_net(self):
        self._images = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        with tf.variable_scope('vgg16'):
            # block 1
            net = conv2d(self._images, 64, 3, scope="conv1_1")
            net = conv2d(net, 64, 3, scope="conv1_2")
            net = max_pool2d(net, scope="pool1")
            # block 2
            net = conv2d(net, 128, 3, scope="conv2_1")
            net = conv2d(net, 128, 3, scope="conv2_2")
            net = max_pool2d(net, scope="pool2")
            # block 3
            net = conv2d(net, 256, 3, scope="conv3_1")
            net = conv2d(net, 256, 3, scope="conv3_2")
            net = conv2d(net, 256, 3, scope="conv3_3")
            net = max_pool2d(net, scope="pool3")
            # block 4
            net = conv2d(net, 512, 3, scope="conv4_1")
            net = conv2d(net, 512, 3, scope="conv4_2")
            net = conv2d(net, 512, 3, scope="conv4_3")
            net = max_pool2d(net, scope="pool4")

            # block 5
            net = conv2d(net, 512, 3, scope="conv5_1")
            net = conv2d(net, 512, 3, scope="conv5_2")
            net = conv2d(net, 512, 3, scope="conv5_3")
            # output
            self._out = net
            net = max_pool2d(net, scope="pool5")
            # block 6
            net = flatten(net)
            print(net)
            net = dense(net, 4096, tf.nn.relu, scope='fc6')
            net = dense(net, 4096, tf.nn.relu, scope='fc7')
            net = dense(net, 1000, None, scope='fc8')

            self.prob = tf.nn.softmax(net, name='prob')

    @property
    def images(self):
        return self._images

    @property
    def out(self):
        return self._out


# Conv2d: for stride = 1


def conv2d(x, filters, kernel_size, stride=1, padding="same",
           dilation_rate=1, activation=tf.nn.relu, scope="conv2d"):
    kernel_sizes = [kernel_size] * 2
    strides = [stride] * 2
    dilation_rate = [dilation_rate] * 2
    return tf.layers.conv2d(x, filters, kernel_sizes, strides=strides,
                            dilation_rate=dilation_rate, padding=padding,
                            name=scope, activation=activation)

# max pool2d: default pool_size = stride


def max_pool2d(x, scope="max_pool"):
    pool_sizes = [2, 2]
    strides = [2, 2]
    return tf.layers.max_pooling2d(x, pool_sizes, strides, name=scope, padding="same")

# pad2d: for conv2d with stride > 1


def pad2d(x, pad):
    return tf.pad(x, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])

# dropout


def dropout(x, rate=0.5, is_training=True):
    return tf.layers.dropout(x, rate=rate, training=is_training)


def dense(x, output, activation=tf.nn.relu, scope="fc"):
    return tf.layers.dense(x, units=output, activation=activation, name=scope)


def flatten(x):
    shape = x.get_shape()
    num_features = shape[1:4].num_elements()
    return tf.reshape(x, [-1, num_features])


def sppnet(x, spatial_pool_size):
    # get feature size
    height = int(x.get_shape()[1])
    width = int(x.get_shape()[2])

    # get batch size
    batch_num = int(x.get_shape()[0])

    for i in range(len(spatial_pool_size)):

        # stride
        stride_h = int(np.ceil(height/spatial_pool_size[i]))
        stride_w = int(np.ceil(width/spatial_pool_size[i]))

        # kernel
        window_w = int(np.ceil(width/spatial_pool_size[i]))
        window_h = int(np.ceil(height/spatial_pool_size[i]))

        # max pool
        max_pool = tf.nn.max_pool(x, ksize=[1, window_h, window_w, 1], strides=[
                                  1, stride_h, stride_w, 1], padding='SAME')

        if i == 0:
            spp = tf.reshape(max_pool, [batch_num, -1])
        else:
            # concat each pool result
            spp = tf.concat(
                axis=1, values=[spp, tf.reshape(max_pool, [batch_num, -1])])

    return spp


if __name__ == "__main__":
    data = np.load(vgg16_npy_path, encoding='latin1').item()
    keys = sorted(data.keys())
    weights = []
    for key in keys:
        weights.append(data[key][0])
        weights.append(data[key][1])
    variables = tf.global_variables()
    for i in range(len(variables)):
        print(i+1)
        sess.run(variables[i].assign(weights[i]))
