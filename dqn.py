# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from gym.wrappers import Monitor
from collections import deque
from tqdm import tqdm
import time

EPISODES=1000



class DQNAgent:
	def __init__(self, model,state_size, action_size):
		self.memory = deque(maxlen=2000)
		self.gamma = 0.95    # discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.state_size=state_size
		self.action_size=action_size
		self.model = model


	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.uniform()  <= self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])  # returns action

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = (reward + self.gamma *
						  np.amax(self.model.predict(next_state)[0]))
			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)

	def fit(self,env,batch_size = 32):
		for e in range(EPISODES):
			state = env.reset()
			state = np.reshape(state, [1, self.state_size])
			tot_reward=0
			for _ in range(500):
				tot_reward+=1
				action = self.act(state)
				next_state, reward, done, _ = env.step(action)	
				reward = reward if not done else -10
				next_state = np.reshape(next_state, [1, self.state_size])
				self.remember(state, action, reward, next_state, done)
				state = next_state
				if done:
					print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, tot_reward, self.epsilon))
					break

			if len(self.memory) > batch_size:
				self.replay(batch_size)
			else:
				print("not now")

	def play(self,env):
		observation = env.reset()
		obs = observation
		state = obs
		done = False
		tot_reward = 0.0
		# video_recorder = VideoRecorder(env, video_path, enabled=video_path is not None)
		# env = Monitor(env, './')

		while not done:
			env.render()
			# env.unwrapped.render()
			# video_recorder.capture_frame()
			state = np.squeeze(state).reshape(1,4)
			action = self.act(state)
			observation, reward, done, info = env.step(action)
			obs = observation
			state = obs    
			tot_reward += reward
		env.close()
		print('Game ended! Total reward: {}'.format(tot_reward))
		# video_recorder.close()
		# video_recorder.enabled = False
		return tot_reward


