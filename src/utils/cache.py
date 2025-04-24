import os
import hashlib
import pickle
import zlib
import time
import threading
import logging
import functools
import inspect
import json
from typing import Any, Dict, Optional, Tuple, List, Union, Callable
from pathlib import Path
import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class IntermediateResultCache:
    """Cache for storing intermediate computation results to avoid redundant work."""
    
    def __init__(self, cache_dir: str = "../cache/intermediate/", max_memory_mb: int = 512,
                 compression_level: int = 6, ttl_seconds: int = 86400):
        """Initialize the intermediate result cache.
        
        Args:
            cache_dir: Directory to store cached results
            max_memory_mb: Maximum memory usage in MB for in-memory cache
            compression_level: Compression level for disk storage
            ttl_seconds: Time-to-live for cached items in seconds (default: 1 day)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.compression_level = compression_level
        self.ttl_seconds = ttl_seconds
        
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._memory_usage = 0
        self._lock = threading.Lock()
        self._stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'memory_stores': 0,
            'disk_stores': 0,
            'memory_evictions': 0,
            'hit_rate': 0.0
        }
        
        logger.info(f"Intermediate result cache initialized with directory: {cache_dir}")
        logger.info(f"Maximum memory usage: {max_memory_mb} MB, TTL: {ttl_seconds}s")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and valid, None otherwise
        """
        with self._lock:
            # Try memory cache first
            if key in self._memory_cache:
                item = self._memory_cache[key]
                
                # Check if item is expired
                if time.time() > item['expiry']:
                    # Remove expired item
                    logger.debug(f"Memory cache item expired: {key}")
                    self._memory_usage -= item['size']
                    del self._memory_cache[key]
                else:
                    # Update access time and return value
                    self._memory_cache[key]['last_access'] = time.time()
                    self._stats['memory_hits'] += 1
                    self._update_hit_rate()
                    logger.debug(f"Memory cache hit: {key}")
                    return item['value']
            
            # Try disk cache
            disk_key = self._hash_key(key)
            cache_file = self.cache_dir / f"{disk_key}.pkl.gz"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        compressed_data = f.read()
                        decompressed_data = zlib.decompress(compressed_data)
                        item = pickle.loads(decompressed_data)
                    
                    # Check if item is expired
                    if time.time() > item['expiry']:
                        # Remove expired item
                        logger.debug(f"Disk cache item expired: {key}")
                        cache_file.unlink(missing_ok=True)
                        self._stats['misses'] += 1
                        self._update_hit_rate()
                        return None
                    
                    # Check if item can fit in memory cache
                    item_size = len(decompressed_data)
                    if self._can_store_in_memory(item_size):
                        # Store in memory for faster access next time
                        self._memory_cache[key] = {
                            'value': item['value'],
                            'expiry': item['expiry'],
                            'last_access': time.time(),
                            'size': item_size
                        }
                        self._memory_usage += item_size
                        logger.debug(f"Moved item from disk to memory cache: {key}, size: {item_size/1024:.2f} KB")
                    
                    self._stats['disk_hits'] += 1
                    self._update_hit_rate()
                    logger.debug(f"Disk cache hit: {key}")
                    return item['value']
                    
                except Exception as e:
                    logger.warning(f"Error loading cache item {key}: {str(e)}")
                    # Remove corrupted cache file
                    cache_file.unlink(missing_ok=True)
            
            # Item not found
            self._stats['misses'] += 1
            self._update_hit_rate()
            logger.debug(f"Cache miss: {key}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store item in cache.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Custom time-to-live in seconds (optional)
        """
        if value is None:
            return
            
        expiry = time.time() + (ttl if ttl is not None else self.ttl_seconds)
        
        try:
            # Serialize to estimate size
            serialized = pickle.dumps(value)
            item_size = len(serialized)
            
            with self._lock:
                # Store in memory if possible
                if self._can_store_in_memory(item_size):
                    # Make room in memory cache if needed
                    self._ensure_memory_available(item_size)
                    
                    # Store in memory
                    self._memory_cache[key] = {
                        'value': value,
                        'expiry': expiry,
                        'last_access': time.time(),
                        'size': item_size
                    }
                    self._memory_usage += item_size
                    self._stats['memory_stores'] += 1
                    logger.debug(f"Stored in memory cache: {key}, size: {item_size/1024:.2f} KB")
                
                # Always store on disk for persistence
                disk_key = self._hash_key(key)
                cache_file = self.cache_dir / f"{disk_key}.pkl.gz"
                
                # Create item for disk
                disk_item = {
                    'key': key,
                    'value': value,
                    'expiry': expiry,
                    'created': time.time()
                }
                
                # Compress and store
                serialized_item = pickle.dumps(disk_item)
                compressed_data = zlib.compress(serialized_item, self.compression_level)
                
                with open(cache_file, 'wb') as f:
                    f.write(compressed_data)
                
                self._stats['disk_stores'] += 1
                logger.debug(f"Stored in disk cache: {key}, " 
                            f"size: {len(compressed_data)/1024:.2f} KB (compressed), "
                            f"original: {item_size/1024:.2f} KB")
                
        except Exception as e:
            logger.warning(f"Error storing cache item {key}: {str(e)}")
    
    def _can_store_in_memory(self, size: int) -> bool:
        """Check if an item can be stored in memory.
        
        Args:
            size: Size of the item in bytes
            
        Returns:
            Whether the item can be stored in memory
        """
        # Check if item itself fits in memory
        if size > self.max_memory_bytes:
            return False
            
        # Check if we have enough space available
        return size <= (self.max_memory_bytes - self._memory_usage) or len(self._memory_cache) == 0
    
    def _ensure_memory_available(self, required_size: int) -> None:
        """Ensure enough memory is available by evicting items if needed.
        
        Args:
            required_size: Required size in bytes
        """
        if required_size <= (self.max_memory_bytes - self._memory_usage):
            # Already enough space
            return
            
        # Sort items by last access time (oldest first)
        items_by_access = sorted(
            [(k, v['last_access'], v['size']) for k, v in self._memory_cache.items()],
            key=lambda x: x[1]
        )
        
        # Evict items until we have enough space
        freed_space = 0
        evicted_count = 0
        
        for key, _, size in items_by_access:
            if (self._memory_usage - freed_space - size) <= (self.max_memory_bytes - required_size):
                # We have enough space now
                break
                
            # Evict this item
            freed_space += size
            del self._memory_cache[key]
            evicted_count += 1
            
        # Update memory usage
        self._memory_usage -= freed_space
        
        if evicted_count > 0:
            self._stats['memory_evictions'] += evicted_count
            logger.debug(f"Evicted {evicted_count} items from memory cache, freed {freed_space/1024:.2f} KB")
    
    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            # Clear memory cache
            self._memory_cache.clear()
            self._memory_usage = 0
            
            # Clear disk cache
            for cache_file in self.cache_dir.glob("*.pkl.gz"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {str(e)}")
            
            # Reset stats
            for key in self._stats:
                self._stats[key] = 0
                
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        with self._lock:
            stats = self._stats.copy()
            stats['memory_usage_mb'] = self._memory_usage / (1024 * 1024)
            stats['memory_items'] = len(self._memory_cache)
            
            # Count disk items
            disk_items = len(list(self.cache_dir.glob("*.pkl.gz")))
            stats['disk_items'] = disk_items
            
            # Calculate disk usage
            disk_usage = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl.gz"))
            stats['disk_usage_mb'] = disk_usage / (1024 * 1024)
            
            return stats
    
    def _update_hit_rate(self) -> None:
        """Update the cache hit rate statistic."""
        total = self._stats['memory_hits'] + self._stats['disk_hits'] + self._stats['misses']
        if total > 0:
            hit_rate = (self._stats['memory_hits'] + self._stats['disk_hits']) / total
            self._stats['hit_rate'] = hit_rate
    
    def _hash_key(self, key: str) -> str:
        """Hash a key to ensure safe filenames and fixed length.
        
        Args:
            key: Original key
            
        Returns:
            Hashed key
        """
        return hashlib.md5(key.encode('utf-8')).hexdigest()
    
    def cleanup_expired(self) -> int:
        """Clean up expired items.
        
        Returns:
            Number of items removed
        """
        removed_count = 0
        current_time = time.time()
        
        with self._lock:
            # Clean memory cache
            memory_expired = [k for k, v in self._memory_cache.items() if current_time > v['expiry']]
            for key in memory_expired:
                self._memory_usage -= self._memory_cache[key]['size']
                del self._memory_cache[key]
                removed_count += 1
            
            # Clean disk cache
            for cache_file in self.cache_dir.glob("*.pkl.gz"):
                try:
                    with open(cache_file, 'rb') as f:
                        compressed_data = f.read()
                        decompressed_data = zlib.decompress(compressed_data)
                        item = pickle.loads(decompressed_data)
                    
                    if current_time > item['expiry']:
                        cache_file.unlink()
                        removed_count += 1
                except Exception:
                    # If we can't read it, remove it
                    cache_file.unlink(missing_ok=True)
                    removed_count += 1
        
        logger.info(f"Cache cleanup removed {removed_count} expired items")
        return removed_count


# Global cache instance for easy access
result_cache = IntermediateResultCache()


def cached_result(ttl: Optional[int] = None, include_args: bool = True):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Custom time-to-live in seconds (optional)
        include_args: Whether to include function arguments in the cache key
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key based on function name and arguments
            if include_args:
                # Create a deterministic representation of args and kwargs
                arg_str = str(args)
                kwarg_str = json.dumps(
                    {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                     for k, v in sorted(kwargs.items())}, 
                    sort_keys=True
                )
                cache_key = f"{func.__module__}.{func.__name__}:{arg_str}:{kwarg_str}"
            else:
                cache_key = f"{func.__module__}.{func.__name__}"
            
            # Try to get from cache
            cached_value = result_cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value
            
            # Compute the result
            logger.debug(f"Cache miss for {func.__name__}, computing result")
            result = func(*args, **kwargs)
            
            # Store in cache
            result_cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


def cached_property(ttl: Optional[int] = None):
    """
    Decorator to create a cached property that is computed once and then cached.
    
    Args:
        ttl: Custom time-to-live in seconds (optional)
        
    Returns:
        Decorated property
    """
    def decorator(func):
        cache_key = f"{func.__module__}.{func.__qualname__}"
        
        @property
        @functools.wraps(func)
        def wrapper(self):
            # Include instance id in cache key to avoid sharing between instances
            instance_cache_key = f"{cache_key}:{id(self)}"
            
            # Try to get from cache
            cached_value = result_cache.get(instance_cache_key)
            if cached_value is not None:
                return cached_value
            
            # Compute the result
            result = func(self)
            
            # Store in cache
            result_cache.set(instance_cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


class numpy_cache:
    """
    Context manager for caching numpy arrays during computation.
    
    This is useful for complex computations that generate multiple intermediate
    numpy arrays that can be reused later.
    
    Usage:
        with numpy_cache("computation_name") as cache:
            # Check if we have cached result
            if "result_key" in cache:
                result = cache["result_key"]
            else:
                # Compute and cache
                result = compute_expensive_function()
                cache["result_key"] = result
    """
    
    def __init__(self, name: str, ttl: Optional[int] = None):
        """
        Initialize numpy cache context manager.
        
        Args:
            name: Name of the computation (used in cache key)
            ttl: Custom time-to-live in seconds (optional)
        """
        self.name = name
        self.ttl = ttl
        self.cache_data = {}
        self.is_loaded = False
    
    def __enter__(self):
        """Enter context manager."""
        # Try to load existing cache
        cached_data = result_cache.get(f"numpy_cache:{self.name}")
        if cached_data is not None:
            self.cache_data = cached_data
            self.is_loaded = True
            logger.debug(f"Loaded numpy cache for {self.name} with {len(cached_data)} items")
        else:
            logger.debug(f"Created new numpy cache for {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and save cache."""
        if exc_type is None:  # Only save if no exception occurred
            result_cache.set(f"numpy_cache:{self.name}", self.cache_data, self.ttl)
            logger.debug(f"Saved numpy cache for {self.name} with {len(self.cache_data)} items")
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self.cache_data
    
    def __getitem__(self, key: str) -> np.ndarray:
        """Get an item from the cache."""
        return self.cache_data[key]
    
    def __setitem__(self, key: str, value: np.ndarray) -> None:
        """Set an item in the cache."""
        self.cache_data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get an item from the cache with a default value."""
        return self.cache_data.get(key, default)


def start_cleanup_thread(interval_seconds: int = 3600) -> threading.Thread:
    """
    Start a background thread to periodically clean up expired cache items.
    
    Args:
        interval_seconds: Interval between cleanup runs in seconds
        
    Returns:
        The started thread
    """
    def cleanup_loop():
        while True:
            time.sleep(interval_seconds)
            try:
                result_cache.cleanup_expired()
            except Exception as e:
                logger.error(f"Error in cache cleanup: {str(e)}")
    
    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()
    logger.info(f"Started cache cleanup thread with interval: {interval_seconds}s")
    return thread


# Start cleanup thread when module is imported
cleanup_thread = start_cleanup_thread() 