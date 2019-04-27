import tensorflow as tf

weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = tf.contrib.layers.l2_regularizer(0.0001)

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x

def global_context_block(x, channels, use_bias=True, sn=False, scope='gc_block'):
    with tf.variable_scope(scope):
        with tf.variable_scope('context_modeling'):
            bs, h, w, c = x.get_shape().as_list()
            input_x = x
            input_x = hw_flatten(input_x)  # [N, H*W, C]
            input_x = tf.transpose(input_x, perm=[0, 2, 1])
            input_x = tf.expand_dims(input_x, axis=1)

            context_mask = conv(x, channels=1, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv')
            context_mask = hw_flatten(context_mask)
            context_mask = tf.nn.softmax(context_mask, axis=1)  # [N, H*W, 1]
            context_mask = tf.transpose(context_mask, perm=[0, 2, 1])
            context_mask = tf.expand_dims(context_mask, axis=-1)

            context = tf.matmul(input_x, context_mask)
            context = tf.reshape(context, shape=[bs, 1, 1, c])

        with tf.variable_scope('transform_0'):
            context_transform = conv(context, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv_0')
            context_transform = layer_norm(context_transform)
            context_transform = relu(context_transform)
            context_transform = conv(context_transform, channels=c, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv_1')
            context_transform = sigmoid(context_transform)

            x = x * context_transform

        with tf.variable_scope('transform_1'):
            context_transform = conv(context, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv_0')
            context_transform = layer_norm(context_transform)
            context_transform = relu(context_transform)
            context_transform = conv(context_transform, channels=c, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv_1')

            x = x + context_transform

        return x

def layer_norm(x, scope='layer_norm'):
    return tf.contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)
def relu(x):
    return tf.nn.relu(x)

def sigmoid(x):
    return tf.sigmoid(x)

def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm