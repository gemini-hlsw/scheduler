# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import json
import redis
import os

from redis import RedisError

REDIS_URL = os.environ.get("REDISCLOUD_URL")


class RedisClient:
    def __init__(self):
        self._redis_client = redis.from_url(REDIS_URL, socket_timeout=600, socket_connect_timeout=30) if REDIS_URL else None
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

    def get_nested_value(self, main_key, key):
        value = self._redis_client.hget(main_key, key)
        return json.loads(value) if value else None

    # Function to set a nested value
    def set_nested_value(self, main_key, key, value):
        self._redis_client.hset(main_key, key, json.dumps(value))

    def get_whole_dict(self, main_key):
        flat_dict = self._redis_client.hgetall(main_key)
        # Convert bytes to str for both keys and values
        flat_dict = {k.decode('utf-8'): v.decode('utf-8') for k, v in flat_dict.items()}
        return RedisClient.unflatten_dict(flat_dict)

    def set_whole_dict(self, main_key, nested_dict, batch_size=100):
        # Flatten and store the dictionary
        flat_dict = RedisClient.flatten_dict(nested_dict)
        pipeline = self._redis_client.pipeline(transaction=False)

        total_items = len(flat_dict)

        for i, (k, v) in enumerate(flat_dict.items(), 1):

            pipeline.hset(main_key, k, json.dumps(v.to_dict()))

            if i % batch_size == 0 or i == total_items:
                try:
                    pipeline.execute()
                    print(f"Processed {i}/{total_items} items")
                except RedisError as e:
                    print(f"Error occurred at item {i}: {str(e)}")
                    # Implement retry logic here if needed
                pipeline = self._redis_client.pipeline(transaction=False)  # Reset pipeline


redis_client = RedisClient()

__all__ = [
    redis_client
]

if __name__ == "__main__":
    ndict = {"obs1": {'2': json.dumps({"visibility_slot_idx": "12-506",
                                       "visibility_time": "200"}),
                     '3': json.dumps({"visibility_slot_idx": "12-506",
                                       "visibility_time": "200"}),
                     '4': json.dumps({"visibility_slot_idx": "12-506",
                                       "visibility_time": "200"}),
                     '5': json.dumps({"visibility_slot_idx": "12-506",
                                       "visibility_time": "200"}),
                     '6': json.dumps({"visibility_slot_idx": "12-506",
                                       "visibility_time": "200"})
                  },
             "obs2": {'2': json.dumps({"visibility_slot_idx": "12-506",
                                       "visibility_time": "200"}),
                      '3': json.dumps({"visibility_slot_idx": "12-506",
                                       "visibility_time": "200"}),
                      '4': json.dumps({"visibility_slot_idx": "12-506",
                                       "visibility_time": "200"}),
                      '5': json.dumps({"visibility_slot_idx": "12-506",
                                       "visibility_time": "200"}),
                      '6': json.dumps({"visibility_slot_idx": "12-506",
                                       "visibility_time": "200"})
                      },
             "obs3": {'2': json.dumps({"visibility_slot_idx": "12-506",
                                       "visibility_time": "200"}),
                      '3': json.dumps({"visibility_slot_idx": "12-506",
                                       "visibility_time": "200"}),
                      '4': json.dumps({"visibility_slot_idx": "12-506",
                                       "visibility_time": "200"}),
                      '5': json.dumps({"visibility_slot_idx": "12-506",
                                       "visibility_time": "200"}),
                      '6': json.dumps({"visibility_slot_idx": "12-506",
                                       "visibility_time": "200"})
                      }
             }
    #flatten = redis_client.flatten_dict(ndict)
    #print(flatten)
    #print(redis_client.unflatten_dict(flatten))
    # redis_client.set_whole_dict(ndict)
    d = redis_client.get_whole_dict()
    print(d)
    #print(redis_client.get_nested_value('obs3:3'))
    #redis_client.set_nested_value('obs3:3',{"visibility_slot_idx": "0-340",
    #                                   "visibility_time": "100"})
    #print(redis_client.get_nested_value('obs3:3'))


