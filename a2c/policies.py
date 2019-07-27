import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, sample

class CnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = (32, 32, 3)
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = 3
        X = tf.placeholder(tf.uint8, ob_shape)
        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope("common", reuse=reuse):
                h = conv(tf.cast(X, tf.float32), 'c1', nf=32, rf=5, stride=1, init_scale=np.sqrt(2), pad="SAME")
                h2 = conv(h, 'c2', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), pad="SAME")
            with tf.variable_scope("pi1", reuse=reuse):
                h3 = conv_to_fc(h2)
                h4 = fc(h3, 'fc1', nh=256, init_scale=np.sqrt(2))
                pi_ = fc(h4, 'pi', nact)
                pi = tf.nn.softmax(pi_)
                vf = fc(h4, 'v', 1)
            with tf.variable_scope("xy0", reuse=reuse):
                pi_xy0_ = conv(h2, 'xy0', nf=1, rf=1, stride=1, init_scale=np.sqrt(2))
                pi_xy0__ = conv_to_fc(pi_xy0_)
                pi_xy0 = tf.nn.softmax(pi_xy0__)
            with tf.variable_scope("xy1", reuse=reuse):
                pi_xy1_ = conv(h2, 'xy1', nf=1, rf=1, stride=1, init_scale=np.sqrt(2))
                pi_xy1__ = conv_to_fc(pi_xy1_)
                pi_xy1 = tf.nn.softmax(pi_xy1__)
        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = []

        def step(ob, *_args, **_kwargs):
            _pi1, _xy0, _xy1, _v = sess.run([pi, pi_xy0, pi_xy1, v0], {X: ob})
            return _pi1, _xy0, _xy1, _v, []

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X: ob})

        self.X = X
        self.pi = pi
        self.pi_xy0 = pi_xy0
        self.pi_xy1 = pi_xy1
        self.vf = vf
        self.step = step
        self.value = value

    def act(self, ob):
        ac, ac_dist, logp = self._act(ob[None])
        return ac[0], ac_dist[0], logp[0]
