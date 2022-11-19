try:
    from .leveldb import TypedLevelDB
except ImportError:
    from .plyvel import TypedLevelDB