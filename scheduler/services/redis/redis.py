import os
import redis

REDIS_URL = os.environ.get("REDISCLOUD_URL")
redis_client = redis.from_url(REDIS_URL) if REDIS_URL else None


# Function to flatten the nested dictionary
def flatten_dict(d, parent_key='', sep=':'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Flatten and store the dictionary
flat_dict = flatten_dict(nested_dict)
for k, v in flat_dict.items():
    r.hset('my_nested_dict', k, json.dumps(v))

# Function to get a nested value
def get_nested_value(key):
    value = r.hget('my_nested_dict', key)
    return json.loads(value) if value else None

# Function to set a nested value
def set_nested_value(key, value):
    r.hset('my_nested_dict', key, json.dumps(value))
if not redis_client:
    raise ValueError("REDISCLOUD_URL env var is not set up correctly.")

__all__ = [
    redis_client
]
