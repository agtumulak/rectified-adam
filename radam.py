"""Custom optimizer classes.
"""
from __future__ import absolute_import, division, print_function

import keras.backend as K
from keras.optimizers import Optimizer
from keras.legacy import interfaces


class RAdam(Optimizer):
    """Rectified Adam optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.

    # References
        - [On the Variance of the Adaptive Learning Rate and Beyond](
           https://arxiv.org/pdf/1908.03265.pdf)
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, **kwargs):
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        learning_rate = kwargs.pop('lr', learning_rate)
        super(RAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.learning_rate
        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr / (1. - K.pow(self.beta_1, t))

        ms = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='m_' + str(i))
              for (i, p) in enumerate(params)]
        vs = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='v_' + str(i))
              for (i, p) in enumerate(params)]

        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            # Compute length of approximated SMA
            rho_inf = 2. / (1. - self.beta_2) - 1.
            rho_t = rho_inf - 2. * t * K.pow(self.beta_2, t) / (1. - K.pow(self.beta_2, t))

            # Decide wheter to apply bias-corrected moving second moment
            if rho_t > 4.:
                # Variance rectification term
                r = K.sqrt(
                        ((rho_t - 4.) * (rho_t - 2.) * rho_inf * (1. - K.pow(self.beta_2, t))
                      / ((rho_inf - 4.) * (rho_inf - 2.) * rho_t) * v_t))
            else:
                r = 1.

            p_t = p - lr_t * r * m_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'epsilon': self.epsilon}
        base_config = super(RAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
