import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import gym
import time


from RunHist import RewardHist


env = gym.make('SkiingDeterministic-v4')
action_size = env.action_space.n


def get_frame_reward(I, prev):
    I = I[:, :, 1]
    I = I[74:75, 8:152]  # Jugador 92, bandera roja 50, bandera azul 72
    if 72 not in I and 50 not in I:
        return 0
    if 72 in I:
        flags = np.where(I == 72)
    else:
        flags = np.where(I == 50)

    player = np.where(I == 92)[1]

    if len(player) == 0:
        return 1

    player = player.mean()

    if len(flags[1]) == 2:
        if player >= flags[1][0] and player <= flags[1][1]:
            return 1
        else:
            return -1
    else:
        return prev

# reward discount used by Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)


def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    # we go from last reward to first one so we don't have to do exponentiations
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            # if the game ended (in Pong), reset the reward sum
            running_add = 0
        # the point here is to use Horner's method to compute those rewards efficiently
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r)  # normalizing the result
    discounted_r /= np.std(discounted_r)  # idem
    return discounted_r


class Skier:
    def __init__(self, gamma=0.95, epsilon=1, e_min=0.05, e_decay=0.99, ideal_flag_interval=25):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = e_min
        self.epsilon_decay = e_decay

        self.episode = 0

        self.ideal_flag_interval = ideal_flag_interval

        self.autosave = None

        self.model = self._make_model()

        self.reset()

    def _make_model(self):
        model = Sequential()
        model.add(Dense(
            units=256,
            input_dim=72*72,
            activation='relu',
            # kernel_initializer='glorot_uniform'
        ))
        model.add(Dense(
            units=128,
            activation='relu'
        ))
        model.add(Dense(
            units=3,
            activation='softmax',
            # kernel_initializer='RandomNormal'
        ))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        return model

    def preprocessFrame(self, I):
        """ 
        Outputs a 72x72 image where background is black
        and important game elements are white.
        Output is [0,1]
        """
        I = I[::2, ::2, 1]
        I = I[31:103, 4:76]
        I[I == 236] = 0
        I[I == 192] = 0
        I[I == 214] = 0
        I[I != 0] = 255
        return I/255

    def decay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def reset(self):
        self.last_reward = 0
        self.rewards = []
        self.train_x = []
        self.train_y = []
        self.lr_counter = 0

    def action(self, frame, training=False):
        frame = self.preprocessFrame(frame).flatten()
        x = np.array([frame])
        probs = self.model.predict(x)
        y = np.random.choice([0, 1, 2], p=probs[0])
        if float('nan') in probs[0]:
            print("NANANANANANANANANANANANANANANANANANA", probs[0])
        if not training:
            return y
        else:
            # Explore a bit
            if np.random.rand() <= self.epsilon:
                y = np.random.choice([0, 1, 2])

        # Append flattened frame to x_train
        self.train_x.append(frame)
        # Append selected action to y_train
        self.train_y.append(to_categorical(y, num_classes=3))
        return y

    def register_frame(self, frame):
        frame_reward = get_frame_reward(frame, self.last_reward)
        reward = 0
        self.lr_counter += 1
        if frame_reward == 0 and self.last_reward != 0:
            reward = self.last_reward
            reward -= 0.5 * \
                np.tanh((0.05 * (self.lr_counter - self.ideal_flag_interval)))
            self.lr_counter = 0
        self.last_reward = frame_reward

        self.rewards.append(reward)

    def train(self, verbose=0):
        sample_weights = discount_rewards(self.rewards, self.gamma)
        self.model.fit(
            x=np.vstack(self.train_x),
            y=np.vstack(self.train_y),
            verbose=verbose,
            sample_weight=sample_weights
        )
        if self.autosave is not None and self.episode % self.autosave == 0:
            self.save("last.h5")
            print("Saved!")
        self.episode += 1
        self.decay()

    def total_reward(self):
        return np.array(self.rewards).sum()

    def set_autosave(self, interval):
        self.autosave = interval

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)


agent = Skier(gamma=0.95, e_decay=0.96)
agent.model.summary()

agent.load("last.h5")

agent.set_autosave(10)
observation = env.reset()
hist = RewardHist(100)
agent.reset()
while True:
    env.render()
    action = agent.action(observation, training=True)

    observation, reward, done, _ = env.step(action)

    agent.register_frame(observation)

    if done:
        total_reward = agent.total_reward()
        hist.add(total_reward)

        if agent.episode % 1 == 0:
            print('# - = - = - = - #')
            print(
                f"Ep: {agent.episode:4}\nTotal reward: {total_reward:.3f}\nEpsilon: {agent.epsilon:.4f}")
            hist.report()

        agent.train(verbose=1)
        agent.reset()

        observation = env.reset()
