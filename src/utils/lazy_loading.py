import importlib
import functools
import logging
import time
import os
import threading
import weakref
from typing import Any, Callable, Dict, List, Optional, Type, Union, TypeVar

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

class LazyLoader:
    """
    Lazily loads modules or classes when they are first accessed.
    
    This helps reduce startup time and memory usage by only loading
    components when they are actually needed.
    """
    
    def __init__(self, module_path: str, class_name: Optional[str] = None,
                 init_params: Optional[Dict[str, Any]] = None):
        """
        Initialize lazy loader.
        
        Args:
            module_path: Import path to the module
            class_name: Name of the class to load from the module (optional)
            init_params: Parameters to pass to the class constructor (optional)
        """
        self.module_path = module_path
        self.class_name = class_name
        self.init_params = init_params or {}
        self._instance = None
        self._module = None
        self._class = None
        self._lock = threading.RLock()  # Reentrant lock to avoid deadlocks
        
    def __call__(self, *args, **kwargs):
        """Create and return an instance of the lazily loaded class."""
        with self._lock:
            if self._instance is None:
                # Combine init params with call params, with call params taking precedence
                combined_params = {**self.init_params, **kwargs}
                
                # Load the class if not already loaded
                if self._class is None:
                    self._load_class()
                    
                # Create instance
                start_time = time.time()
                self._instance = self._class(*args, **combined_params)
                load_time = time.time() - start_time
                
                logger.info(f"Created instance of {self.class_name} in {load_time:.2f}s")
                
            return self._instance
    
    def _load_class(self) -> None:
        """Load the module and class."""
        start_time = time.time()
        
        # Load module if not already loaded
        if self._module is None:
            try:
                self._module = importlib.import_module(self.module_path)
                logger.debug(f"Loaded module {self.module_path}")
            except ImportError as e:
                logger.error(f"Failed to import module {self.module_path}: {e}")
                raise
        
        # Get class from module
        if self.class_name:
            try:
                self._class = getattr(self._module, self.class_name)
                logger.debug(f"Loaded class {self.class_name} from {self.module_path}")
            except AttributeError as e:
                logger.error(f"Failed to get class {self.class_name} from {self.module_path}: {e}")
                raise
        else:
            # If no class name specified, use the module itself
            self._class = self._module
        
        load_time = time.time() - start_time
        logger.debug(f"Loaded {self.module_path}.{self.class_name} in {load_time:.2f}s")
    
    def clear(self) -> None:
        """Clear the cached instance to free memory."""
        with self._lock:
            self._instance = None
            logger.debug(f"Cleared instance of {self.class_name}")


class LazyLoadedComponent:
    """
    Base class for components that should be lazily loaded.
    
    Components inheriting from this class will only be fully initialized
    when their methods are first called.
    """
    
    def __init__(self):
        """Initialize with delayed initialization."""
        self._initialized = False
        self._initializing = False
        self._lock = threading.RLock()
        self._lazy_init_done = False
        
    def _ensure_initialized(self) -> None:
        """Ensure the component is initialized."""
        if self._lazy_init_done:
            return
            
        with self._lock:
            if not self._initialized and not self._initializing:
                try:
                    self._initializing = True
                    start_time = time.time()
                    
                    # Call the initialization method
                    self._lazy_init()
                    
                    # Mark as initialized
                    self._initialized = True
                    self._lazy_init_done = True
                    
                    init_time = time.time() - start_time
                    logger.debug(f"Lazy initialized {self.__class__.__name__} in {init_time:.2f}s")
                finally:
                    self._initializing = False
    
    def _lazy_init(self) -> None:
        """
        Initialize the component's resources.
        
        This method should be overridden by subclasses to perform
        the actual initialization.
        """
        pass


class LazyInitializedMethod:
    """Decorator for methods that should trigger lazy initialization."""
    
    def __init__(self, method: Callable):
        """Initialize with the method to decorate."""
        self.method = method
        functools.update_wrapper(self, method)
        
    def __get__(self, obj: Any, objtype=None) -> Callable:
        """Get method implementation that ensures initialization."""
        if obj is None:
            return self.method
            
        @functools.wraps(self.method)
        def wrapper(*args, **kwargs):
            # Ensure initialized before calling method
            if isinstance(obj, LazyLoadedComponent):
                obj._ensure_initialized()
            return self.method(obj, *args, **kwargs)
            
        return wrapper


def lazy_initialized(method: Callable) -> LazyInitializedMethod:
    """
    Decorator for methods in LazyLoadedComponent classes.
    
    Methods decorated with this will ensure the component is initialized
    before the method is called.
    
    Args:
        method: Method to decorate
        
    Returns:
        Decorated method
    """
    return LazyInitializedMethod(method)


class ResourceManager:
    """
    Manages resources to ensure they are released when no longer needed.
    
    This class keeps track of resource-intensive components and releases
    them when memory pressure is high or they haven't been used for a while.
    """
    
    def __init__(self, max_unused_time: int = 300, memory_threshold: float = 0.8):
        """
        Initialize resource manager.
        
        Args:
            max_unused_time: Maximum time in seconds a resource can remain unused
            memory_threshold: Memory usage threshold (0.0-1.0) to trigger cleanup
        """
        self.resources: Dict[str, Dict[str, Any]] = {}
        self.max_unused_time = max_unused_time
        self.memory_threshold = memory_threshold
        self._lock = threading.RLock()
        
        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"ResourceManager initialized (max unused time: {max_unused_time}s, "
                   f"memory threshold: {memory_threshold:.1%})")
    
    def register(self, name: str, resource: Any, cleanup_func: Optional[Callable[[Any], None]] = None) -> None:
        """
        Register a resource for management.
        
        Args:
            name: Name of the resource
            resource: The resource object
            cleanup_func: Function to call when releasing the resource
        """
        with self._lock:
            # Store resource with metadata
            self.resources[name] = {
                'resource': weakref.ref(resource),  # Store weak reference to allow GC
                'cleanup_func': cleanup_func,
                'last_access': time.time(),
                'access_count': 0
            }
            logger.debug(f"Registered resource: {name}")
    
    def get(self, name: str) -> Optional[Any]:
        """
        Get a registered resource.
        
        Args:
            name: Name of the resource
            
        Returns:
            The resource if found, None otherwise
        """
        with self._lock:
            if name in self.resources:
                # Get resource from weak reference
                resource_ref = self.resources[name]['resource']
                resource = resource_ref()
                
                if resource is not None:
                    # Update access time and count
                    self.resources[name]['last_access'] = time.time()
                    self.resources[name]['access_count'] += 1
                    return resource
                else:
                    # Resource was garbage collected
                    logger.debug(f"Resource {name} was garbage collected")
                    del self.resources[name]
                    
            return None
    
    def release(self, name: str) -> bool:
        """
        Explicitly release a resource.
        
        Args:
            name: Name of the resource
            
        Returns:
            Whether the resource was successfully released
        """
        with self._lock:
            if name in self.resources:
                return self._release_resource(name)
            return False
    
    def release_all(self) -> None:
        """Release all managed resources."""
        with self._lock:
            for name in list(self.resources.keys()):
                self._release_resource(name)
    
    def _release_resource(self, name: str) -> bool:
        """
        Release a resource by name.
        
        Args:
            name: Name of the resource
            
        Returns:
            Whether the resource was successfully released
        """
        if name not in self.resources:
            return False
            
        resource_info = self.resources[name]
        resource = resource_info['resource']()
        
        if resource is not None and resource_info['cleanup_func'] is not None:
            try:
                # Call cleanup function
                resource_info['cleanup_func'](resource)
                logger.debug(f"Released resource: {name}")
            except Exception as e:
                logger.error(f"Error releasing resource {name}: {e}")
        
        # Remove from managed resources
        del self.resources[name]
        return True
    
    def _cleanup_loop(self) -> None:
        """Background thread that periodically cleans up unused resources."""
        while True:
            time.sleep(60)  # Check every minute
            
            try:
                self._check_resources()
            except Exception as e:
                logger.error(f"Error in resource cleanup: {e}")
    
    def _check_resources(self) -> None:
        """Check resources and release unused ones if needed."""
        with self._lock:
            # Get current time
            current_time = time.time()
            
            # Check memory usage
            try:
                import psutil
                memory_usage = psutil.virtual_memory().percent / 100.0
                memory_pressure = memory_usage > self.memory_threshold
            except ImportError:
                memory_pressure = False
            
            # List resources to release
            to_release = []
            
            for name, info in self.resources.items():
                # Check if resource is unused for too long
                elapsed = current_time - info['last_access']
                
                # Release if unused for too long or if under memory pressure
                if elapsed > self.max_unused_time or (memory_pressure and elapsed > 60):
                    to_release.append(name)
            
            # Release resources
            for name in to_release:
                logger.info(f"Auto-releasing unused resource: {name} "
                           f"(unused for {current_time - self.resources[name]['last_access']:.0f}s)")
                self._release_resource(name)


# Global resource manager instance
resource_manager = ResourceManager()


def lazy_import(module_path: str, class_name: Optional[str] = None, 
               init_params: Optional[Dict[str, Any]] = None) -> LazyLoader:
    """
    Create a lazy loader for a module or class.
    
    Args:
        module_path: Import path to the module
        class_name: Name of the class to load from the module (optional)
        init_params: Parameters to pass to the class constructor (optional)
        
    Returns:
        Lazy loader instance
    """
    return LazyLoader(module_path, class_name, init_params)


def managed_resource(name: Optional[str] = None, cleanup_func: Optional[Callable[[Any], None]] = None,
                    max_unused_time: Optional[int] = None):
    """
    Decorator to register a class instance as a managed resource.
    
    Args:
        name: Name of the resource (defaults to class name)
        cleanup_func: Function to call when releasing the resource
        max_unused_time: Maximum time in seconds a resource can remain unused
        
    Returns:
        Decorated class
    """
    def decorator(cls):
        original_init = cls.__init__
        
        @functools.wraps(original_init)
        def init_wrapper(self, *args, **kwargs):
            # Call original init
            original_init(self, *args, **kwargs)
            
            # Register as managed resource
            resource_name = name or cls.__name__
            instance_name = f"{resource_name}_{id(self)}"
            
            # Create cleanup function
            def cleanup(instance):
                if hasattr(instance, 'cleanup') and callable(instance.cleanup):
                    instance.cleanup()
                elif hasattr(instance, 'close') and callable(instance.close):
                    instance.close()
                elif hasattr(instance, 'release') and callable(instance.release):
                    instance.release()
            
            # Use provided cleanup function or default
            final_cleanup = cleanup_func or cleanup
            
            # Register with resource manager
            if max_unused_time is not None:
                # Create custom resource manager with different timeout
                custom_manager = ResourceManager(max_unused_time=max_unused_time)
                custom_manager.register(instance_name, self, final_cleanup)
            else:
                # Use global resource manager
                resource_manager.register(instance_name, self, final_cleanup)
        
        cls.__init__ = init_wrapper
        return cls
    
    return decorator 