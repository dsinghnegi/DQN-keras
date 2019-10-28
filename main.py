import dqn
import gym
# from gym.monitoring import VideoRecorder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Flatten

import numpy as np

def build_model(state_size,action_size,learning_rate):
	model = Sequential()
	model.add(Dense(24, input_dim=state_size, activation='relu'))
	# model.add(Dense(64, activation='relu'))
	# model.add(Dense(32, activation='relu'))
	model.add(Dense(24, activation='relu'))
	model.add(Dense(action_size, activation='linear'))
	model.compile(loss='mse',optimizer=Adam(lr=learning_rate))
	return model

def main():
	env = gym.make('CartPole-v1')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n

	learning_rate=1e-3
	model=build_model(state_size,action_size, learning_rate)
	agent = dqn.DQNAgent(model, state_size, action_size)

	agent.fit()

	env = gym.wrappers.Monitor(env, "./video", video_callable=lambda episode_id: True,force=True)
	for _ in range(10):
		agent.play(env)







if __name__ == '__main__':
	main()