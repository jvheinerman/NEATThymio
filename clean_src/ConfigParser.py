import json

# Reads the values contained in the passed configuration file and stores them in this object
class ConfigParser(object):
    def __init__(self, filename):
        json_data = open(filename)
        data = json.load(json_data)
        json_data.close()
        self.__address = data["address"]
        self.__port = data["port"]
        self.__bots = data["bots"]

    @property
    def address(self):
        return self.__address

    @property
    def port(self):
        return self.__port

    @property
    def bots(self):
        return self.__bots
