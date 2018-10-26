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

import json
try:
    from jsmin import jsmin
except ModuleNotFoundError:
    jsmin = lambda x: x

_sentinel = object()


def load_params(filepath):
    """
    Load your hyper parameters from a json file.
    :param filepath: Path to the json file.
    :return: A hyper parameters object.
    """
    with open(filepath) as file:
        return HyperParams(json.loads(jsmin(file.read())))


class HyperParams(object):
    """
    Converts a dictionary into an object.
    """
    def __init__(self, d=None):
        """
        Create an object from a dictionary.

        :param d: The dictionary to convert.
        """
        if d is not None:
            for a, b in d.items():
                if isinstance(b, (list, tuple)):
                    setattr(self, a, [HyperParams(x) if isinstance(x, dict) else x for x in b])
                else:
                    setattr(self, a, HyperParams(b) if isinstance(b, dict) else b)

    def to_dict(self):
        return dict((key, value.to_dict()) if isinstance(value, HyperParams) else (key, value)
                    for (key, value) in self.__dict__.items())

    def __repr__(self):
        return "HyperParams(" + self.__str__() + ")"
    
    def __str__(self):
        return json.dumps(self.to_dict(), indent=4, sort_keys=True)

    def get(self, key, default=_sentinel):
        """
        Get the value specified in the dictionary or a default.
        :param key: The key which should be retrieved.
        :param default: The default that is returned if the key is not set.
        :return: The value from the dict or the default.
        """
        if default is _sentinel:
            default = HyperParams({})
        return self.__dict__[key] if key in self.__dict__ else default

    def __getitem__(self, key):
        """
        Get the value specified in the dictionary or a dummy.
        :param key: The key which should be retrieved.
        :return: The value from the dict or a dummy.
        """
        return self.get(key)

    #def __getattr__(self, attr):
    #    print("Warning hyperparameter {} not found returning dummy object.".format(attr))
    #    return HyperParams()
