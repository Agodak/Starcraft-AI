import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe


class PolicyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh)
        self.dense2 = tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)

    def call(self, inputs):
        result = self.dense1(inputs)
        result = self.dense2(result)
        return result


class ValueModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh)
        self.dense2 = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        result = self.dense1(inputs)
        result = self.dense2(result)
        return result


class PolicyGradientMemory():
    def __init__(self, gamma):
        self.memory = []
        self.gamma = gamma

    def reset(self):
        self.memory = []

    def append(self, transition):
        self.memory.append(transition)

    def sample(self):
        obs, rewards, dones, available_actions, army_counts = zip(self.memory)
        s = np.array(list(obs))
        a = np.array(list(available_actions))
        s1 = np.array(list(army_counts))
        r = np.array(list(rewards), dtype="float32")
        done = np.array(list(dones))

        r = np.expand_dims(r, axis=1)
        return [s, a, s1, r, done]

    def __len__(self):
        return len(self.memory)

    def __str__(self):
        result = []
        for i in range(self.__len__()):
            result.append(self.memory[i].__str__() + " \n")
        return "".join(result)


class Agent():
    def __init__(self, env, lr, discount):
        self.env = env
        self.episode_durations = []
        self.episode_loss = []
        self.gamma = discount
        self.policy_model = PolicyModel()
        self.value_model = ValueModel()
        self.memory = PolicyGradientMemory(discount)
        self.policy_optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
        self.value_optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)

    def getAction(self, s):
        s = tf.convert_to_tensor([s], dtype=tf.float32)
        action_probability = self.policy_model(s).numpy()
        action = np.random.choice([0, 1], p=action_probability[0])
        return action

    def learn(self, episodes, regularization, value_gradient):
        for i in range(episodes):
            s = self.env.reset()
            steps = 0
            while True:
                action = self.getAction(s)
                s_1, reward, done, info = self.env.step(action)
                self.memory.append((s, action, s_1, reward, done))
                s = s_1
                steps += 1
                if done:
                    break
            s, a, s1, r, d = self.memory.sample()
            s = tf.constant(s, dtype=tf.float32)
            a = tf.one_hot(a, depth=2, dtype=tf.int32)
            s1 = tf.constant(s1, dtype=tf.float32)
            r = tf.constant(r, dtype=tf.float32)
            d = tf.expand_dims(tf.constant(1 - d, dtype=tf.float32), axis=1)
            v = self.value_model(s) * value_gradient
            v_prime = self.value_model(s1) * value_gradient
            q = r + self.gamma * d * v_prime
            Adv = q - v
            with tfe.GradientTape() as tape:
                v = self.value_model(s)
                loss = (q - v) ** 2
                loss_value = tf.reduce_mean(loss)
            grads = tape.gradient(loss_value, self.value_model.variables)
            self.value_optimizer.apply_gradients(zip(grads, self.value_model.variables),
                                                 global_step=tf.train.get_or_create_global_step())
            with tfe.GradientTape() as tape:
                action_probability = self.policy_model(s)
                loss = action_probability * tf.cast(a, dtype="float32")
                loss = tf.reduce_sum(loss, reduction_indices=1)
                loss = tf.log(loss)
                loss_value = - tf.reduce_mean(loss * (Adv)) * regularization
            grads = tape.gradient(loss_value, self.policy_model.variables)
            self.policy_optimizer.apply_gradients(zip(grads, self.policy_model.variables),
                                                  global_step=tf.train.get_or_create_global_step())
            self.memory.reset()
            self.episode_durations.append(steps)

    def run(self, env):
        self.env = env
        s = self.env.reset()
        steps = 0
        while True:
            self.env.render()
            action = self.getAction(s)
            s_1, reward, done, info = self.env.step(action)
            s = s_1
            steps += 1
            if done:
                print("Episode finished successfully after {} timesteps".format(steps))
                break
        self.env.close()