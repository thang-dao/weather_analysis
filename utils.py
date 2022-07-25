import json


def read_json(filename):
    with open("{}".format(filename), "r") as fp:
        content = json.load(fp)
    return content