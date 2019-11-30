# Singleton/SingletonMetaClass.py
import threading



# Adapted from https://github.com/tomerghelber/singleton-factory
class Singleton(object):
    """
    Singleton Factory - keeps one object with the same hash
        of the same cls.
    Returns:
        An existing instance.
    """

    instance = None
    lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        with self.__class__.lock:
            if not self.__class__.instance:
                self.__class__.instance = object.__init__(self, *args, **kwargs)
