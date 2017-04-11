from __future__ import absolute_import, division, print_function

from collections import deque
from itertools import count

import distributed


class DaskClient(object):
    """ A simple wrapper around ``distributed.Client`` to conform to our API"""
    def __init__(self, client):
        self._client = client
        self._as_completed = distributed.as_completed([], with_results=True)

    def submit(self, func, *args):
        future = self._client.submit(func, *args)
        self._as_completed.add(future)
        return future

    def has_results(self):
        return not self._as_completed.queue.empty()

    def next_batch(self, block=False):
        return self._as_completed.next_batch(block=block)


class SerialClient(object):
    """ A simple client to run in serial.

    This queues work until ``max_queue_size`` tasks have been submitted,
    then it returns results one at a time.

    """
    def __init__(self):
        self._queue = deque()
        # For now, we use unique integers to mock future objects.  We don't
        # do anything fancy with them.  We only use them as keys in a dict.
        self._counter = count()

    def submit(self, func, *args):
        future = next(self._counter)
        self._queue.append((future, func, args))
        return future

    def has_results(self):
        return False

    def next_batch(self, block=False):
        if not block:
            return ()
        future, func, args = self._queue.popleft()
        result = func(*args)
        return ((future, result),)

