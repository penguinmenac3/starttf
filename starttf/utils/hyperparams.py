from starttf.utils.dict2obj import json_file_to_object


def load_params(filepath):
    """
    Load your hyper parameters from a json file.
    :param filepath: Path to the json file.
    :return: A hyper parameters object.
    """
    return json_file_to_object(filepath)
