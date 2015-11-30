import math
import tensorflow as tf

def unbiased_linear(input_tensor, input_size, output_size, name=""):
    stddev = 1.0/math.sqrt(input_size)
    with tf.name_scope(name) as scope:
        weights = tf.Variable(tf.truncated_normal([input_size, output_size], stddev = stddev, name="weights"))
        return tf.matmul(input_tensor, weights)

def moments(x, axes, name=None):
  with tf.op_scope([x, axes], name, "moments"):
    x = tf.convert_to_tensor(x, name="x")
    divisor = tf.constant(1.0)
    for d in xrange(len(x.get_shape())):
      if d in axes:
        divisor *= tf.to_float(tf.shape(x)[d])
    divisor = tf.inv(divisor, name="divisor")
    axes = tf.constant(axes, name="axes")
    mean = tf.mul(tf.reduce_sum(x, axes), divisor, name="mean")
    var = tf.mul(tf.reduce_sum(tf.square(x - mean), axes),
                       divisor, name="variance")
    return mean, var

def batch_normalization(batch, mean=None, var=None):
    if mean == None or var == None:
        mean, var = moments(batch, axes=[0])
    return (batch - mean) / (var + tf.constant(1e-10))

def encode(inputs, noise_std, layer_sizes, L):
    wi = lambda inits, size, name: inits * tf.Variable(tf.ones([size]), name=name)
    stack = []
    noise = tf.random_normal([layer_sizes[0]]) * noise_std
    h = inputs + noise
    for l in range(L):
        z_pre = unbiased_linear(h, layer_sizes[l], layer_sizes[l+1])
        m, v = moments(z_pre, axes=[0])
        z = batch_normalization(z_pre, m, v)
        noise = tf.random_normal([layer_sizes[l+1]]) * noise_std
        if l != L-1:
            h = tf.nn.relu(z_pre + noise + wi(0.0, layer_sizes[l+1], 'beta'+str(l)))
        else:
            h = wi(1.0, layer_sizes[l+1], 'gamma') * (z_pre + noise + wi(0.0, layer_sizes[l+1], 'beta'+str(l)))
        stack.append({'z': z, 'm': m, 'v': v, 'h': h})
    return h, stack

def decode(stack_clean, stack_corrupted, layer_sizes, L):
    def g(z_c, u, size):
        wi = lambda inits, name: inits * tf.Variable(tf.ones([size]), name=name)
        sigval = wi(0.0, 'c1') + wi(1.0, 'c2') * z_c + wi(0.0, 'c3') * u + wi(0.0, 'c4') * z_c * u
        sigval = tf.sigmoid(sigval)
        z_est = wi(0.0, 'a1') + wi(1.0, 'a2') * z_c + wi(0.0, 'a3') * u + wi(0.0, 'a4') * z_c * u + wi(1.0, 'b1') * sigval
        return z_est

    cost = 0
    for l in range(L, 0, -1):
        t = stack_corrupted[l-1]
        z_c, m_c, v_c, h_c = t['z'], t['m'], t['v'], t['h']
        z = stack_clean[l-1]['z']
        if l == L:
            u = batch_normalization(h_c)
        else:
            u_pre = unbiased_linear(z_est, layer_sizes[l+1], layer_sizes[l])
            u = batch_normalization(u_pre)
        z_est = g(z_c, u, layer_sizes[l])
        z_est_bn = batch_normalization(z_est, m_c, v_c)
        div = tf.to_float(tf.constant(layer_sizes[l]) * tf.shape(z)[0])
        cost = cost + tf.reduce_sum(tf.nn.l2_normalize(z_est_bn - z, 1)) / div
    return cost

def model(inputs, outputs, layer_sizes):
    L = len(layer_sizes) - 1
    h_c, stack_corrupted = encode(inputs, tf.constant(0.3), layer_sizes, L)
    h, stack_clean = encode(inputs, tf.constant(0.0), layer_sizes, L)
    u_cost = decode(stack_clean, stack_corrupted, layer_sizes, L)
    y = h_c
    cost = 0
    N = tf.shape(outputs)
    y_N = tf.slice(h_c, [0, 0], N)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_N, outputs, name='xentropy')
    cost = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    cost = cost + u_cost
    return y, cost
