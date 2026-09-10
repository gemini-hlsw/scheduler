# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause


__all__ = [
    'Singleton',
    'AsyncSingleton'
]

import asyncio
import threading
from typing import Dict, Any


class Singleton(type):
    """
    Thread-safe Singleton metaclass.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in Singleton._instances:
            Singleton._instances[cls] = super().__call__(*args, **kwargs)
        return Singleton._instances[cls]


class AsyncSingleton(type):
    """
    Thread-safe async-compatible Singleton metaclass.
    Each class using this metaclass will have its own singleton instance.
    """
    _instances: Dict[type, Any] = {}
    _locks: Dict[type, asyncio.Lock] = {}
    _thread_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        """Prevent direct instantiation."""
        raise TypeError(
            f"Cannot instantiate {cls.__name__} directly. "
            f"Use 'await {cls.__name__}.instance()' instead."
        )

    async def instance(cls, *args, **kwargs):
        """
        Async factory method to get or create the singleton instance.
        Thread-safe and async-safe with double-checked locking.
        """
        # First check without lock (fast path)
        if cls not in cls._instances:
            # Ensure we have an asyncio lock for this class
            with cls._thread_lock:
                if cls not in cls._locks:
                    cls._locks[cls] = asyncio.Lock()

            # Async lock for the actual instance creation
            async with cls._locks[cls]:
                # Double-check: another coroutine might have created it
                if cls not in cls._instances:
                    # Create instance by bypassing __call__
                    instance = super(AsyncSingleton, cls).__call__(*args, **kwargs)

                    # If the instance has a connect method, call it
                    if hasattr(instance, 'connect') and callable(getattr(instance, 'connect')):
                        await instance.connect()

                    cls._instances[cls] = instance

        return cls._instances[cls]
