# MIT License
# 
# Copyright (c) 2018 Michael Fuerst
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

class Agent(object):
    def __init__(self, env, model):
        """
        An agent to train a model for reinforcement learning.
        :param env: The gym style environment for training.
        :param model: A model object that inherits from starttf.models.model.RLModel.
        """
        self.env = env
        self.model = model

    def act(self, state):
        """
        Act on a state to receive the picked action.
        """
        raise NotImplementedError()

    def step(self, **kwargs):
        """
        One step of the learning loop.

        A step typically involves doing an action in the environment and receiving a reward.
        The reward is then send to the model as a training signal.

        Typically does the following:
        1. Get State
        2. Pick action
        3. Simulate action on env
        4. Get reward and next state
        5. Update Model (self.model.update(...))
        """
        raise NotImplementedError()

    def reset(self):
        """
        This methods resets the env of the agent.

        When overwriting, call super method.
        """
        self.env.reset()

    def learn(self, steps=1, **kwargs):
        """
        Train the model using the environment and the agent.

        Note that the model might be shared between multiple agents (which most probably are of the same type)
        at the same time.
        :param steps: The number of steps to train for.
        """
        # TODO add some housekeeping
        for i in range(steps):
            self.step(**kwargs)
