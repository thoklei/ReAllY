import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Dense, Input, Lambda)


class A2C(Model):
    def __init__(self, layers, action_dim, state_dim):
        super(A2C, self).__init__()
        self.mu_layer = [
            tf.keras.layers.Dense(
                units=num_units,
                activation='relu',
                name=f'Policy_mu_{i}'
                ) for i, num_units in enumerate(layers)]

        self.readout_mu = tf.keras.layers.Dense(units=action_dim,
                                                activation=None,
                                                name='Policy_mu_readout'
                                                )

        self.sigma_layer = [
            tf.keras.layers.Dense(
                units=num_units,
                activation='relu',
                name=f'Policy_sigma_{i}'
                ) for i, num_units in enumerate(layers)]

        self.readout_sigma = tf.keras.layers.Dense(units=action_dim,
                                                   activation=None,
                                                   name='Policy_sigma_readout'
                                                   )

        self.value_layer = [
            tf.keras.layers.Dense(
                units=num_units,
                activation='relu',
                name=f'Value_layer_{i}'
                ) for i, num_units in enumerate(layers)]

        self.readout_value = tf.keras.layers.Dense(units=1,
                                                   activation=None,
                                                   name='Value_readout'
                                                   )

    @tf.function
    def call(self, input_state):
        output = {}
        mu_pred = input_state
        sigma_pred = input_state
        value_pred = input_state
        for layer in self.mu_layer:
            mu_pred = layer(mu_pred)
        for layer in self.sigma_layer:
            sigma_pred = layer(sigma_pred)
        for layer in self.value_layer:
            value_pred = layer(value_pred)

        # Actor
        output["mu"] = tf.squeeze(self.readout_mu(mu_pred))
        output["sigma"] = tf.squeeze(tf.abs(self.readout_sigma(sigma_pred)))
        # Critic
        output["value_estimate"] = tf.squeeze(self.readout_value(value_pred))
        return output


class TargetNetwork(Model):

    def __init__(self, layers, k, state_dim):
        """
        Constructor for the RND-target network.

        layers = list of layer sizes
        k = dimensionality of the embedding space
        """

        super(TargetNetwork, self).__init__()

        self.rnd_layer = [
            tf.keras.layers.Dense(
                units=num_units,
                activation='relu') for i, num_units in enumerate(layers)]

        self.readout = tf.keras.layers.Dense(units=k, activation=None)


    @tf.function
    def call(self, input_state):

        embedding = input_state

        for layer in self.rnd_layer:
            embedding = layer(embedding)

        readout = self.readout(embedding)

        return readout

# def dense_block(X, units_list, activation, name=None):
#     for i, units in enumerate(units_list):
#         if name is None:
#             X = Dense(units, activation=activation)(X)
#         else:
#             X = Dense(units, activation=activation, name=name.format(i))(X)
#     return X
#
#
# class A2C(Model):
#     def __init__(self, layers, action_dim, state_dim):
#         inp = Input((state_dim,))
#         mu = dense_block(inp, layers, 'relu', "Policy_mu_{}")
#         mu_out = Dense(action_dim, activation=None, name="Policy_mu_readout")(mu)
#
#         sigma = dense_block(inp, layers, 'relu', "Policy_sigma_{}")
#         sigma_out = Dense(units=action_dim, activation=None, name='Policy_sigma_readout')(sigma)
#
#         value = dense_block(inp, layers, 'relu', "Value_layer_{}")
#         value_out = Dense(units=1, activation=None, name='Value_readout')(value)
#
#         out = Lambda(
#             lambda outputs: {
#                 "mu": tf.squeeze(outputs[0]),
#                 "sigma": tf.squeeze(tf.abs(outputs[1])),
#                 "value_estimate": tf.squeeze(outputs[2])
#             }
#         )([mu_out, sigma_out, value_out])
#
#         super(A2C, self).__init__(inp, out)
#
#
# class TargetNetwork(Model):
#
#     def __init__(self, layers, k, state_dim):
#         """
#         Constructor for the RND-target network.
#
#         layers = list of layer sizes
#         k = dimensionality of the embedding space
#         """
#         inp = Input((state_dim,))
#         X = dense_block(inp, layers, "relu", "")
#         out = Dense(units=k, activation=None, name="output")(X)
#
#         super(TargetNetwork, self).__init__(inp, out)

@tf.function
def train_on_batch_ppo(
        agent, state_batch, action_batch, advantage_batch, log_prob_batch, returns_batch,
        ppo_clipping_value: float, critic_discount: float, entropy_beta: float,
        critic_loss_fn, ppo_optimizer
):
    """ This function performs a single training step on one batch of data for a ppo agent. """
    with tf.GradientTape() as tape:
        # Old policy
        old_log_prob = log_prob_batch
        # New policy
        new_log_prob, entropy = agent.flowing_log_prob(state_batch, action_batch, return_entropy=True)
        # using log rules to compute the ratio: A/B = exp(ln(A) - ln(B))
        ratio = tf.exp(new_log_prob - tf.cast(old_log_prob, tf.float32))

        ppo1 = ratio * tf.expand_dims(advantage_batch, 1)
        ppo2 = tf.clip_by_value(ratio, 1 - ppo_clipping_value, 1 + ppo_clipping_value) * tf.expand_dims(advantage_batch, 1)

        actor_loss = -tf.reduce_mean(tf.minimum(ppo1, ppo2), 0)

        value_target = returns_batch
        value_pred = agent.v(state_batch)
        critic_loss = critic_loss_fn(value_target, value_pred)

        total_loss = actor_loss + critic_discount * critic_loss - entropy_beta * entropy

    gradients = tape.gradient(total_loss, agent.model.trainable_variables)

    ppo_optimizer.apply_gradients(zip(gradients, agent.model.trainable_variables))

    return actor_loss, critic_loss, total_loss


@tf.function
def train_on_batch_rnd(
    state_batch, target_network, predictor_network, pred_optimizer
):
    """ This function performs a single training step on one batch of data for RND. """
    # Obtain so called features from the target network which the predictor needs to imitate.
    target_pred = target_network(state_batch)

    with tf.GradientTape() as tape:
        feature_pred = predictor_network(state_batch)
        rnd_loss = tf.keras.losses.MSE(target_pred, feature_pred)

    rnd_gradients = tape.gradient(rnd_loss, predictor_network.trainable_variables)
    pred_optimizer.apply_gradients(zip(rnd_gradients, predictor_network.trainable_variables))
    return rnd_loss
