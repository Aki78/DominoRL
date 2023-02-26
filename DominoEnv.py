import numpy as np
import gym
from gym import spaces
from torch import nn


class DominoEnv(gym.Env):
    def __init__(self):
        self.tiles = np.array([[i, j] for i in range(7) for j in range(i, 7)])
        self.player1_tiles = None
        self.player2_tiles = None
        self.board = []
        self.current_player = None
        self.done = False

        self.action_space = spaces.Discrete(28)
        self.observation_space = spaces.Box(low=0, high=1, shape=(56,))

    def reset(self):
        self.player1_tiles = np.random.choice(self.tiles, size=7, replace=False)
        self.player2_tiles = np.random.choice(np.setdiff1d(self.tiles, self.player1_tiles), size=7, replace=False)
        self.board = []
        self.current_player = 0
        self.done = False
        return self.get_observation()

    def get_observation(self):
        obs = np.zeros(56)
        for i, tile in enumerate(self.player1_tiles):
            obs[self.tile_to_index(tile, i)] = 1
        for i, tile in enumerate(self.player2_tiles):
            obs[self.tile_to_index(tile, i+7)] = 1
        for i, tile in enumerate(self.board):
            obs[self.tile_to_index(tile, i+14)] = 1
        obs[54] = self.current_player
        obs[55] = len(self.board)
        return obs

    def step(self, action):
        if self.current_player == 0:
            tiles = self.player1_tiles
        else:
            tiles = self.player2_tiles

        tile = self.index_to_tile(action)
        if tile in tiles:
            if len(self.board) == 0:
                self.board.append(tile)
            else:
                last_tile = self.board[-1]
                if tile[0] == last_tile[1]:
                    self.board.append(tile)
                elif tile[1] == last_tile[1]:
                    self.board.append([tile[1], tile[0]])
                else:
                    if self.current_player == 0:
                        self.current_player = 1
                    else:
                        self.current_player = 0
                    return self.get_observation(), -1, False, {}

            if self.current_player == 0:
                self.player1_tiles = np.delete(self.player1_tiles, np.where((self.player1_tiles == tile).all(axis=1))[0][0], axis=0)
            else:
                self.player2_tiles = np.delete(self.player2_tiles, np.where((self.player2_tiles == tile).all(axis=1))[0][0], axis=0)

            if len(tiles) == 0:
                self.done = True
                return self.get_observation(), 1, self.done, {}

            if self.current_player == 0:
                self.current_player = 1
            else:
                self.current_player = 0

            return self.get_observation(), 0, self.done, {}

        else:
            return self.get_observation(), -1, self.done, {}

    def tile_to_index(self, tile, idx):
        return tile[0] * 7 + tile[1] + idx * 7

    def index_to_tile(self, index):
        return [index // 7, index % 7]

