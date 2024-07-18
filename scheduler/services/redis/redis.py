# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import json
import redis
import os

REDIS_URL = os.environ.get("REDISCLOUD_URL")


class RedisClient:
    def __init__(self):
        self._redis_client = redis.from_url(REDIS_URL) if REDIS_URL else None
        if not self._redis_client:
            raise ValueError("REDISCLOUD_URL env var is not set up correctly.")

    @staticmethod
    def flatten_dict(d, parent_key='', sep=':'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(RedisClient.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def unflatten_dict(d, sep=':'):
        result = {}
        for key, value in d.items():
            parts = key.split(sep)
            d = result
            for part in parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = json.loads(value)
        return result

    def get_nested_value(self, key):
        value = self._redis_client.hget('my_nested_dict', key)
        return json.loads(value) if value else None

    # Function to set a nested value
    def set_nested_value(self, key, value):
        self._redis_client.hset('my_nested_dict', key, json.dumps(value))

    def get_whole_dict(self):
        flat_dict = self._redis_client.hgetall('my_nested_dict')
        # Convert bytes to str for both keys and values
        flat_dict = {k.decode('utf-8'): v.decode('utf-8') for k, v in flat_dict.items()}
        return RedisClient.unflatten_dict(flat_dict)

    def set_whole_dict(self, nested_dict):
        # Flatten and store the dictionary
        flat_dict = RedisClient.flatten_dict(nested_dict)
        for k, v in flat_dict.items():
            self._redis_client.hset('my_nested_dict', k, json.dumps(v))


redis_client = RedisClient()

__all__ = [
    redis_client
]