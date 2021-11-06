import gym
import tensorflow as tf
from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu
            extracted_features = self.processed_obs
            pi_h = extracted_features
            pi_h = tf.layers.flatten(pi_h)
            pi_h = activ(tf.layers.dense(pi_h, 128, name='pi_fc1'))
            pi_h = tf.layers.dropout(pi_h, 0.25)
            pi_h = activ(tf.layers.dense(pi_h, 64, name='pi_fc2'))
            pi_h = activ(tf.layers.dense(pi_h, 32, name='pi_fc3'))
            pi_latent = pi_h
            vf_h = extracted_features
            vf_h = tf.layers.flatten(vf_h)
            vf_h = activ(tf.layers.dense(vf_h, 128, name='vf_fc1'))
            vf_h = tf.layers.dropout(vf_h, 0.25)
            vf_h = activ(tf.layers.dense(vf_h, 64, name='vf_fc2'))
            vf_h = activ(tf.layers.dense(vf_h, 32, name='vf_fc3'))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
        self._value_fn = value_fn
        self._setup_init()
    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp
    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})