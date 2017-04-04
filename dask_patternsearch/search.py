from __future__ import absolute_import, division, print_function

import itertools
import math
import distributed
import numpy as np

from time import time
from toolz import concat, cons, take

from .stencil import RightHandedSimplexStencil


class Point(object):
    __slots__ = 'point', 'halvings', 'parent', 'is_accepted', 'start_time', 'stop_time', 'result'

    def __init__(self, point, halvings):
        self.point = point
        self.halvings = halvings
        self.is_accepted = False

    def __hash__(self):
        return hash(self.point.tostring())

    def __eq__(self, other):
        return np.array_equal(self.point, other.point)

    def __repr__(self):
        return type(self).__name__ + repr(self.point)[len('array'):]


def randomize_chunk(i, it):
    L = list(take(i, it))
    np.random.shuffle(L)
    return L


def randomize_stencil(dims, it):
    return concat(randomize_chunk(i, it) for i in itertools.count(2*dims, dims))


def search(client, func, x, stepsize, queue_size=None, min_queue_size=None, min_new_submit=0, randomize=False, max_stencil_size=None, stopratio=0.01, max_tasks=None, max_time=None):
    # bound=None, low_memory_stencil=False
    if queue_size is None:
        ncores = client.ncores()
        queue_size = sum(ncores.values()) + len(ncores)
    if min_queue_size is None:
        min_queue_size = queue_size // 2
    if max_stencil_size is None:
        max_stencil_size = 1e9
    dims = len(stepsize)
    max_halvings = math.frexp(1 / stopratio)[1]
    stencil = RightHandedSimplexStencil(dims, max_halvings)
    gridsize = stepsize / 2.**max_halvings

    def to_grid(x):
        return np.round(x / gridsize) * gridsize

    orientation = np.ones(dims)
    cur_point = Point(to_grid(x), -1)
    cur_cost = result = np.inf
    new_point = None
    is_finished = False

    if max_time is not None:
        end_time = time() + max_time
    results = {}
    running = {}
    processing = []
    next_point = None
    next_cost = None

    # Begin from initial point
    future = client.submit(func, cur_point.point)
    as_completed = distributed.as_completed([future], with_results=True)
    running[future] = cur_point
    results[cur_point] = None
    is_contraction = True

    while not is_finished or running or next_point is not None:
        if max_time is not None and time() > end_time:
            is_finished = True

        # Initialize new point
        if new_point is not None or is_contraction:
            if is_contraction:
                cur_point.halvings += 1
                cur_point.parent = cur_point
                is_contraction = False
            else:
                # TODO: if parent.is_accepted and gparent.is_accepted and parent.cost < gparent.cost: DOUBLE
                # TODO: also look at (2*new_point - prev_point) and double as appropriate
                cur_point = new_point
                cur_cost = result
                new_point = None
            cur_stepsize = to_grid(orientation * stepsize / 2.**cur_point.halvings)
            cur_added = 0
            contract_conditions = set()
            it = stencil.generate_stencil_points()
            if randomize:
                it = randomize_stencil(dims, it)
            stencil_points = enumerate(it, 1)
            stencil_index = 0
            if cur_point.halvings >= max_halvings:
                is_finished = True

        # Fill task queue with trial points while waiting for results
        if not is_finished:
            while (
                len(running) < queue_size
                and stencil_index < max_stencil_size
                and (
                    len(running) < min_queue_size
                    or cur_added < min_new_submit
                    or next_point is None and as_completed.queue.empty()
                )
            ):
                try:
                    stencil_index, step = next(stencil_points)
                except StopIteration:
                    if stencil_index < 2 * dims:
                        raise
                    # else warn
                    stencil_index = max_stencil_size = stencil_index
                    break
                if (
                    cur_added >= min_new_submit
                    and stencil_index > 2 * dims
                    and not contract_conditions
                ):
                    is_contraction = True
                    break
                halvings = step.halvings + cur_point.halvings
                if halvings > max_halvings:
                    continue
                trial_point = to_grid(cur_point.point + step.point * cur_stepsize)
                # TODO: check boundary constraints here
                # if check_feasible is not None and not check_feasible(trial_point):
                #     continue
                trial_point = Point(trial_point, halvings)
                has_result = results.get(trial_point, False)
                if stencil_index <= 2 * dims and (has_result is False or has_result is None):
                    contract_conditions.add(trial_point)
                if has_result is False:
                    trial_point.parent = cur_point
                    trial_point.start_time = time()
                    future = client.submit(func, trial_point.point)
                    as_completed.add(future)
                    running[future] = trial_point
                    results[trial_point] = None
                    cur_added += 1
                    if max_tasks is not None and len(results) >= max_tasks:
                        is_finished = True
                        break
            if is_contraction:
                continue

        # Collect all completed tasks, or wait for one if nothing else to do
        block = (
            len(running) >= queue_size
            or (
                next_point is None
                and (is_finished or stencil_index >= max_stencil_size)
            )
        )
        for future, result in as_completed.next_batch(block=block):
            point = running.pop(future)
            point.stop_time = time()
            if next_point is None:
                next_point = point
                next_cost = result
            elif result < next_cost:
                processing.append((next_point, next_cost))
                next_point = point
                next_cost = result
            else:
                processing.append((point, result))

        # Process all results
        # Be greedy: the new point will be the result with the lowest cost.
        # It's possible one could want a different policy to better explore
        # around multiple minima.
        if next_point is not None and (cur_added >= min_new_submit or stencil_index >= max_stencil_size):
            for point, result in cons((next_point, next_cost), processing):
                results[point] = result
                point.result = result
                if result < cur_cost:
                    point.is_accepted = True
                    # The stencil prefers negative directions, so this is the correct sign
                    diff = point.point - cur_point.point
                    orientation = np.where(diff, np.copysign(orientation, diff), orientation)
                    new_point = point
                elif new_point is not None and not is_finished:
                    if point in contract_conditions:
                        contract_conditions.remove(point)
                    if not contract_conditions and stencil_index > 2 * dims:
                        is_contraction = True
            next_point = None
            next_cost = None
            processing.clear()
    return cur_point, results

