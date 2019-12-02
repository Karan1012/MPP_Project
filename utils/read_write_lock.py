class FifoReadWriteLock:

    def __init__(self):
        self.read_acquires = 0
        self.read_releases = 0
        self.writer = False