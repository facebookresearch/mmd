# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import wraps
import time
import copy


class TimeLog:
    def __init__(self, time_log, debug=False):
        self.time_log = time_log
        self.debug = debug

    def add_time_log(self, name, start, end):
        time_log_entry = {'name' : name, 'execution_time': end - start }
        self.time_log.append(time_log_entry)
        if self.debug:
            print("{}: {:2.4f}s".format(time_log_entry['name'], time_log_entry['execution_time']))

    def __str__(self):
        return "\n".join(map(lambda time_log_entry:
            "{}: {:2.4f}s".format(time_log_entry['name'], time_log_entry['execution_time']),
            self.time_log
            ))

_time_log_container = []
TIMING_DEBUG=False
_time_log = TimeLog(_time_log_container, TIMING_DEBUG)

def get_time_log():
    global _time_log, _time_log_container
    time_log = copy.deepcopy(_time_log)
    # Reset time log entries
    _time_log_container = []
    _time_log = TimeLog(_time_log_container, TIMING_DEBUG)
    return time_log


def time_log(f):
    @wraps(f)
    def wrap(*args, **kw):
        start = time.perf_counter()
        result = f(*args, **kw)
        end = time.perf_counter()
        name = f.__module__ + "/" + f.__name__
        _time_log.add_time_log(name, start, end)
        return result
    return wrap
