import json
from jsmin import jsmin


def json_file_to_object(filepath):
    """
    Read a json file directly to an object.
    :param filepath: The filepath which to load.
    :return: The object.
    """
    with open(filepath) as file:
        return Dict2Obj(json.loads(jsmin(file.read())))


class Dict2Obj(object):
    """
    Converts a dictionary into an object.
    """
    def __init__(self, d):
        """
        Create an object from a dictionary.

        :param d: The dictionary to convert.
        """
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [Dict2Obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, Dict2Obj(b) if isinstance(b, dict) else b)

    def to_dict(self):
        return dict((key, value.to_dict()) if isinstance(value, Dict2Obj) else (key, value)
                    for (key, value) in self.__dict__.items())

    def pretty_print(self):
        print(json.dumps(self.to_dict(), indent=4, sort_keys=True))

    def get_or_default(self, key, default):
        """
        Get the value specified in the dictionary or a default.
        :param key: The key which should be retrieved.
        :param default: The default that is returned if the key is not set.
        :return: The value from the dict or the default.
        """
        return self.__dict__[key] if key in self.__dict__ else default

    def get_or_dummy(self, key):
        """
        Get the value specified in the dictionary or a dummy.
        :param key: The key which should be retrieved.
        :return: The value from the dict or a dummy.
        """
        return self.__dict__[key] if key in self.__dict__ else Dict2Obj({})