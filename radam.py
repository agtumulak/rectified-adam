import tensorflow as tf

class RectifiedAdam(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999):
        super().__init__("RectifiedAdam")
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('decay', self._initial_decay)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')

    def _resource_apply_dense(self, grad, var):
        dtype = var.dtype.base_dtype
        lr = self._decayed_lr(dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1 = self._get_hyper('beta_1', dtype)
        beta_2 = self._get_hyper('beta_2', dtype)
        t = tf.cast(self.iterations + 1, dtype)
        beta_1_pow = tf.pow(beta_1, t)
        beta_2_pow = tf.pow(beta_2, t)

        rho_inf = 2. / (1. - beta_2) - 1.
        rho_t = rho_inf - 2. * t * beta_2_pow / (1.0 - beta_2_pow)

        m = m.assign(beta_1 * m + (1. - beta_1) * grad)
        m_hat = m / (1. - beta_1_pow)

        r = tf.sqrt(
                (rho_t - 4.) / (rho_inf - 4.)
              * (rho_t - 2.) / (rho_inf - 2.)
              * rho_inf / rho_t)

        v = v.assign(beta_2 * v + (1. - beta_2) * tf.square(grad))
        v_hat = tf.sqrt(v / (1.0 - beta_2_pow))
        factor = tf.where(rho_t > 4., r / v_hat, 1.0)
        var_update = var.assign_sub(lr * factor * m_hat)

        return tf.group(var_update, m, v)


    def get_config(self):
        config = super().get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'decay': self._serialize_hyperparameter('decay'),})
        return config
