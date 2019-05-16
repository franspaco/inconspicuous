
import numpy as np
import gym
import time
import keyboard

env = gym.make('SkiingDeterministic-v4')
action_size = env.action_space.n



def preprocessFrame(I):
    I = I[57:203, 8:152, 1]
    return I/255

def to_cat(y):
    a = np.zeros(3)
    a[y] = 1
    return a

obs = env.reset()
prev = np.zeros((146, 144))

x = []
y = []

while True:
    action = 0
    if keyboard.is_pressed('a'):
        action = 2
    if keyboard.is_pressed('d'):
        action = 1
    if keyboard.is_pressed('space'):
        break
    
    obs = preprocessFrame(obs)
    data = np.zeros((146, 144, 2))
    data[:, :, 0] = prev
    data[:, :, 1] = obs

    x.append(data)
    y.append(to_cat(action))
    
    obs, reward, done, _ = env.step(action)

    time.sleep(0.05)

    env.render()

    if done:
        obs = env.reset()


X = np.array(x)
Y = np.array(y)

np.save('X.npy', X)
np.save('Y.npy', Y)
