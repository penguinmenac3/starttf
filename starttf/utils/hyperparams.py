from starttf.utils.dict2obj import json_file_to_object


def load_params(filepath):
    return json_file_to_object(filepath)
