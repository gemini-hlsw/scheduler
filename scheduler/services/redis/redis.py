import os
import redis

REDIS_URL = os.environ.get("REDISCLOUD_URL")
redis_client = redis.from_url(REDIS_URL) if REDIS_URL else None

if not redis_client:
    raise ValueError("REDISCLOUD_URL env var is not set up correctly.")

__all__ = [
    redis_client
]