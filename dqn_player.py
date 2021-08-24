
import numpy as np
import random
from gamestate import GameState, Turn
from gamestate import Direction
import tensorflow as tf
from tensorflow import keras
from collections import deque
#tf.enable_eager_execution() # Since we're using TF 1.x, in 2.x this is enabled by default
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Turn off tensorflow-gpu warnings.

class SkipConv2D(keras.layers.Layer):
    def __init__(self, prev_layer_activation, filters, kernel_size, strides, padding, activation):
        super().__init__()
        self.add_layer = keras.layers.Add()
        #self.prev_layer_activation = DQNPlayer.lrelu if prev_layer_activation == 'lrelu' else prev_layer_activation
        self.prev_layer_activation_name = prev_layer_activation
        self.prev_layer_activation = keras.layers.Activation(prev_layer_activation)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.conv_layer = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                              padding=padding, activation='linear', kernel_initializer=keras.initializers.Zeros())
        self.activation_name = activation
        self.activation = keras.layers.Activation(activation)

    def get_config(self):
        return {'prev_layer_activation': self.prev_layer_activation_name,
                'filters': self.filters, 'kernel_size': self.kernel_size, 'strides': self.strides,
                'padding': self.padding, 'activation': self.activation_name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs):
        next_out = self.conv_layer(self.prev_layer_activation(inputs))
        added = self.add_layer([next_out, inputs])
        out = self.activation(added)
        return out

class DQNPlayer:
    # if we load a model we might need this custom activation function
    @staticmethod
    def lrelu(x):
        return tf.keras.activations.relu(x, alpha=0.1)
    
    def __init__(self, replay_buffer_size=20000):
        self.model = None
        self.channels=7
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.replay_weights = deque(maxlen=replay_buffer_size)
        self.replay_alpha = 0.8 # probability of sampling is proportional to w^alpha, where w is replay weight.

        custom_object = {
            'lrelu': DQNPlayer.lrelu,
        }
        keras.utils.get_custom_objects().update(custom_object)

    def create_model(self, board_size):
        # dqn.train(batch_size=1, episodes=3000, max_steps_per_game=30, discount_factor=0.97, optimizer=keras.optimizers.Adam(lr=0.0002))
        """worked with 2 hidden dense layers only
            keras.layers.Dense(20, activation=lrelu),
            keras.layers.Dense(30, activation=lrelu),
            keras.layers.Dense(3, activation='linear')
        """
        self.board_size = board_size

        self.model = keras.models.Sequential([
            keras.layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='same', activation=DQNPlayer.lrelu,
                                input_shape=[board_size, board_size, self.channels]),
            keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation=DQNPlayer.lrelu),
            keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation=DQNPlayer.lrelu),
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation=DQNPlayer.lrelu),
            keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='linear'),
            SkipConv2D(prev_layer_activation=DQNPlayer.lrelu,
                      filters=128, kernel_size=3, strides=1, padding='same', activation=DQNPlayer.lrelu),
            #keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
            #keras.layers.Conv2D(filters=80, kernel_size=3, strides=1, padding='valid', activation='relu'),
            keras.layers.Flatten(),
            #keras.layers.Dense(64, activation=DQNPlayer.lrelu),
            #keras.layers.Dense(30, activation=lrelu),
            keras.layers.Dense(3, activation='linear')
        ])

    def load_model(self, filename, board_size):
        self.board_size = board_size
        self.model = keras.models.load_model(filename, custom_objects={'lrelu': DQNPlayer.lrelu, 'SkipConv2D': SkipConv2D})

    def save_model(self, filename):
        self.model.save(filename)

    def sample_experiences(self, batch_size):
        #indices = np.random.randint(len(self.replay_buffer), size=batch_size)
        """
        indices = [0] * batch_size
        rands = np.random.rand(batch_size)
        
        for i in range(batch_size):
            w = 0.0
            j = -1
            while w < rands[i] * self.replay_cum_weight and j < len(self.replay_buffer):
                j += 1
                w += self.replay_weights[j]
            indices[i] = max(0, j)
        """
        rands = np.random.rand(batch_size)
        cumsum = np.cumsum(self.replay_weights)
        indices = np.clip(np.searchsorted(cumsum, rands * cumsum[-1]), 0, len(self.replay_weights))
        indices = indices.tolist()
            
        batch = [self.replay_buffer[i] for i in indices]
        gses, gses_arrays, actions, rewards, next_gses, next_gses_arrays, endeds = [
            np.array([experience[i] for experience in batch]) for i in range(7)
        ] # each experience has 5 fields
        return gses, gses_arrays, actions, rewards, next_gses, next_gses_arrays, endeds, indices


    def play_one_step(self, gs, epsilon=0.05):
        assert not gs.has_ended, "Game has ended, can't play another step."
        action = self.epsilon_greedy_policy(gs, epsilon)
        next_gs = gs.make_move(action)
        reward = next_gs.score - gs.score
        ended = 1 if next_gs.has_ended else 0
        self.replay_buffer.append((gs, self.gs_to_input(gs), action, reward, next_gs, self.gs_to_input(next_gs), ended))
        
        self.replay_weights.append(2.0) # relatively high weight so it gets sampled at least once.
        
        return next_gs

    def train_minibatch(self, batch, discount_factor=0.20, optimizer=keras.optimizers.Adam(lr=1e-3),
                      loss_fn=keras.losses.mean_squared_error, verbose=False):
        """ Warning: this function hasn't been updated to use prioritized experience replay. """
        gses, gses_arrays, actions, rewards, next_gses, next_gses_arrays, endeds = batch
        next_Q_values = self.model.predict(next_gses_arrays)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = rewards + (1 - endeds) * discount_factor * max_next_Q_values
        mask = tf.one_hot(actions, 3)

        if verbose:
            print(self.model.predict(gses_arrays), target_Q_values, actions)
            #for layer in self.model.layers:
            #    print(layer.get_config(), layer.get_weights())
            #weights_before = self.model.layers[1].get_weights().copy()
            
        with tf.GradientTape() as tape:
            all_Q_values = self.model(gses_arrays, training=True)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(loss_fn(target_Q_values[...,np.newaxis], Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        if verbose:
            #print(keras.backend.eval(grads[0]))
            print(f"all_Q_values: {Q_values}, target_Q_values: {target_Q_values}, actions={actions}")
            print(keras.backend.eval(loss))
            #for layer in self.model.layers:
            #    print(layer.get_config(), layer.get_weights())
            #weights_after = self.model.layers[1].get_weights().copy()
            print(self.model.predict(gses_arrays), target_Q_values, actions)
            print(weights_after[0] - weights_before[0])

    def training_step(self, batch_size=32, discount_factor=0.20, optimizer=keras.optimizers.Adam(lr=1e-3),
                      loss_fn=keras.losses.mean_squared_error, verbose=False):
        gses, gses_arrays, actions, rewards, next_gses, next_gses_arrays, endeds, indices = self.sample_experiences(batch_size)

        best_next_actions = np.argmax(self.model.predict(next_gses_arrays), axis=1)
        next_action_mask = tf.one_hot(best_next_actions, 3).numpy()
        
        next_Q_values = self.target_network.predict(next_gses_arrays)
        #max_next_Q_values = np.max(next_Q_values, axis=1)
        max_next_Q_values = (next_Q_values * next_action_mask).sum(axis=1)
        target_Q_values = rewards + (1 - endeds) * discount_factor * max_next_Q_values
        mask = tf.one_hot(actions, 3)

        if verbose:
            print(self.model.predict(gses_arrays), target_Q_values, actions)
            #for layer in self.model.layers:
            #    print(layer.get_config(), layer.get_weights())
            #weights_before = self.model.layers[1].get_weights().copy()

        with tf.GradientTape() as tape:
            all_Q_values = self.model(gses_arrays, training=True)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(loss_fn(target_Q_values[...,np.newaxis], Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        TD = tf.math.abs(tf.math.subtract(target_Q_values[...,np.newaxis], Q_values)).numpy().reshape(-1)
        TD_weighted = np.power(TD, self.replay_alpha)
        for i in range(len(indices)):
            self.replay_weights[indices[i]] = TD_weighted[i]
            alpha = 0.98
            for t in range(1,10):
                j = indices[i] - t
                if j >= 0:
                    self.replay_weights[j] = max(self.replay_weights[j], self.replay_weights[i]*alpha)
                alpha *= 0.98
                
        if verbose:
            #print(keras.backend.eval(grads[0]))
            print(f"all_Q_values: {Q_values}, target_Q_values: {target_Q_values}, actions={actions}")
            print(keras.backend.eval(loss))
            #for layer in self.model.layers:
            #    print(layer.get_config(), layer.get_weights())
            #weights_after = self.model.layers[1].get_weights().copy()
            print(self.model.predict(gses_arrays), target_Q_values, actions)
            print(weights_after[0] - weights_before[0])

    def train(self, batch_size=32, episodes=500, clone_episodes=100, score_max_len=300, games_to_score=10,
              max_steps_per_game=20, eps=None, **kwargs):
        
        self.target_network = keras.models.clone_model(self.model)
        
        for episode in range(episodes):
            #if episode % 50 == 0:
            #    print(f"Episode: {episode}")
            if episode % 50 == 0:
                print(f"Score: {self.score(games_to_score, max_game_len=score_max_len)}")
            if episodes % clone_episodes == 0:
                self.target_network.set_weights(self.model.get_weights())
            gs = GameState(board_size=self.board_size)
            for step in range(max_steps_per_game):
                if eps is None:
                    epsilon = max(1 - episode / (0.8 * episodes), 0.05)
                else:
                    epsilon = eps
                gs = self.play_one_step(gs, epsilon)
                
                if gs.has_ended:
                    break
                if episode > 25:
                    self.training_step(batch_size, **kwargs)

    def play_full_game(self, max_game_len=100, return_all_states=True, avg_rots=False):
        gs = GameState(board_size=self.board_size)
        if return_all_states:
            states = [gs]
        steps = 0
        while not gs.has_ended and steps < max_game_len:
            steps += 1
            gs = gs.make_move(self.best_move(gs, avg_rots=avg_rots))
            if return_all_states:
                states.append(gs)
        if return_all_states:
            return states
        else:
            return gs

    

    def gamestates_to_animation(self, states, save_filename=None, fps=1):
        """ Returns an animation of the sequence of states.
            save_filename: if filename given, saves gif file to the filename.
            To view a played out game in a Jupyter notebook run:
                from IPython.display import HTML
                HTML(dqn.gamestates_to_animation(dqn.play_full_game()).to_html5_video())
        """
        import matplotlib.pyplot as plt
        from matplotlib import animation
        fig, ax = plt.subplots()
        anim = animation.FuncAnimation(fig, lambda i: states[i].plot(ax), frames=len(states), interval=1000//fps, blit=False)
        if save_filename is not None:
            writer = animation.PillowWriter(fps=fps)
            anim.save(save_filename, writer=writer)
        return anim

    def score(self, n_simulations, max_game_len=200, avg_rots=False):
        res = []
        for i in range(n_simulations):
            gs = self.play_full_game(max_game_len, return_all_states=False, avg_rots=avg_rots)
            res.append(len(gs.snake))
        import math
        return np.mean(res), np.std(res) / math.sqrt(n_simulations)
        
    def epsilon_greedy_policy(self, gs, epsilon=0.05):
        """ Returns a Turn according to following the epsilon greedy policy. """
        if random.random() < epsilon:
            return random.choice([Turn.LEFT, Turn.RIGHT, Turn.STRAIGHT])
        else:
            return self.best_move(gs)

    def model_output(self, gs):
        """ Outputs Q-values for moves [Turn.LEFT, Turn.RIGHT, Turn.STRAIGHT]. """
        return self.model.predict(self.gs_to_input(gs)[np.newaxis,...])
    
    def best_move(self, gs, avg_rots=False):
        """ average_rots: Rotate the board and evaluate the model, then average the results. """
        output = self.model_output(gs)
        if avg_rots:
            for i in range(3):
                gs = gs.rotate_90()
                output += self.model_output(gs)
            output /= 4
        return [Turn.LEFT, Turn.RIGHT, Turn.STRAIGHT][np.argmax(output)]

    def gs_to_input(self, gs: GameState):
        n = gs.board_size
        inputs = np.zeros((n, n, self.channels), dtype='float32')

        # We'll store the food location, snake and the parts of the snake moving in each of the 4 directions
        if gs.food:
            inputs[gs.food[0], gs.food[1], 0] = 1.0

        for (i, j) in gs.snake:
            inputs[i, j, 1] = 1.0

        inputs[gs.snake[0][0], gs.snake[0][1], 2] = 1.0 # head of snake

        dirs = [Direction.LEFT.value, Direction.RIGHT.value, Direction.UP.value, Direction.DOWN.value]
        for k in range(0, len(gs.snake)):
            if k == 0:
                dir_ = gs.direction.value
            else:
                dir_ = (gs.snake[k-1][0] - gs.snake[k][0], gs.snake[k-1][1] - gs.snake[k][1])

            inputs[gs.snake[k][0], gs.snake[k][1], 3 + dirs.index(dir_)] = 1.0
        
        return inputs
