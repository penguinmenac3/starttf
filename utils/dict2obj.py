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
