"""
Usage:

remote = DynamicConfigFile(file_name="kompil_remote.json")
handle_group = remote.handle("my_group")
handle_value = handle_group.handle("my_value", "default")
my_value = handle.read()
print("my_value:", handle.read())
"""
import os
import json


def _modified_date(file_path):
    return os.path.getmtime(file_path)


class BaseDynamicConfig:
    def __init__(self):
        self.__handles = {}

    def handle(self, key, init_value=None):
        handle = self.__handles.get(key)
        if handle is not None:
            return handle
        handle = DynamicConfigHandler(self, key=key, init_value=init_value)
        self.__handles[key] = handle
        return handle

    def _get_cache_key(self, key: str, default=None) -> dict:
        raise NotImplementedError()

    def _set_cache_key(self, key, value):
        raise NotImplementedError()

    def print(self):
        raise NotImplementedError()


class DynamicConfigFile(BaseDynamicConfig):
    def __init__(self, file_name: str):
        super().__init__()
        self.__file_path = os.path.join("/tmp", file_name)
        # Init the file
        self.__cache = {}
        self.__full_write(self.__cache)

    def __full_write(self, content):
        with open(self.__file_path, "w+") as f:
            f.write(json.dumps(content, sort_keys=True, indent=2))
        self.__last_date = _modified_date(self.__file_path)

    def __get_data(self) -> dict:
        # Update cache
        curr_date = _modified_date(self.__file_path)
        if curr_date != self.__last_date:
            with open(self.__file_path, "r") as f:
                self.__cache = json.load(f)
            self.__last_date = _modified_date(self.__file_path)
        # Get value
        return self.__cache

    def _get_cache_key(self, key: str, default=None) -> dict:
        return self.__get_data().get(key, default)

    def _set_cache_key(self, key, value):
        self.__cache[key] = value
        self.__full_write(self.__cache)

    def print(self):
        print(self.__get_data())

    def __str__(self):
        return f"DynamicConfigFile<{self.__file_path}>"


class DynamicConfigHandler(BaseDynamicConfig):
    def __init__(self, parent, key: str, init_value=None):
        super().__init__()
        self.__parent = parent
        self.__key = key
        self.write(init_value)

    def _get_cache_key(self, key: str, default=None) -> dict:
        return self.__parent._get_cache_key(self.__key).get(key, default)

    def _set_cache_key(self, key, value):
        cache = self.__parent._get_cache_key(self.__key)
        if not isinstance(cache, dict):
            cache = {}
        cache[key] = value
        self.__parent._set_cache_key(self.__key, cache)

    def read(self, default=None):
        return self.__parent._get_cache_key(self.__key, default)

    def write(self, value):
        self.__parent._set_cache_key(self.__key, value)

    def print(self):
        self.__parent.print()

    def __str__(self):
        return f'{self.__parent}["{self.__key}"]'
