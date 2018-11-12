import numpy as np
from gym import spaces
from gym import Env


def random_odd(max_index):
    return np.random.randint(0, max_index // 2 + max_index % 2) * 2 + 1

def random_even(max_index):
    return np.random.randint(0, max_index // 2 + 1) * 2


class PointEnv(Env):
    """
    point mass on a 2-D plane
    goals are sampled randomly from a square
    """

    def __init__(self, num_tasks=1, granularity=None):
        self.granularity = granularity
        self.reset_task()
        self.reset()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))


    def reset_task(self, is_evaluation=False):
        '''
        sample a new task randomly

        Problem 3: make training and evaluation goals disjoint sets
        if `is_evaluation` is true, sample from the evaluation set,
        otherwise sample from the training set
        '''
        #====================================================================================#
        #                           ----------PROBLEM 3----------
        #====================================================================================#
        # YOUR CODE HERE
        if self.granularity is not None:
            side_size = self.granularity
            index_max = 20 // side_size - 1

            i = np.random.randint(0, index_max + 1)
            if is_evaluation:
                if i % 2 == 0:
                    j = random_odd(index_max)
                else:
                    j = random_even(index_max)
            else:
                if i % 2 == 0:
                    j = random_even(index_max)
                else:
                    j = random_odd(index_max)

            x = np.random.uniform(-10 + j * side_size, -10 + (j + 1) * side_size)
            y = np.random.uniform(-10 + i * side_size, -10 + (i + 1) * side_size)
            self._goal = np.array([x, y])

        else:
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(-10, 10)
            self._goal = np.array([x, y])

    def reset(self):
        self._state = np.array([0, 0], dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        return np.copy(self._state)

    def reward_function(self, x, y):
        return - (x ** 2 + y ** 2) ** 0.5

    def step(self, action):
        x, y = self._state
        # compute reward, add penalty for large actions instead of clipping them
        x -= self._goal[0]
        y -= self._goal[1]
        # check if task is complete
        done = abs(x) < .01 and abs(y) < .01
        reward = self.reward_function(x, y)
        # move to next state
        self._state = self._state + action
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

    def seed(self, seed):
        np.random.seed = seed
