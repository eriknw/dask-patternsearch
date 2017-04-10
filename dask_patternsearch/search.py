from __future__ import absolute_import, division, print_function

import itertools
import math
import distributed
import numpy as np

from time import time
from toolz import concat, take

from .stencil import RightHandedSimplexStencil


class Point(object):
    __slots__ = 'point', 'halvings', 'parent', 'is_accepted', 'start_time', 'stop_time', 'result'

    def __init__(self, point, halvings):
        self.point = point
        self.halvings = halvings
        self.is_accepted = False
        self.stop_time = None

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


def search(client, func, x0, stepsize, args=(), max_queue_size=None, min_queue_size=None, min_new_submit=0, randomize=True, max_stencil_size=None, stopratio=0.01, max_tasks=None, max_time=None, integer_dimensions=None):
    """ Perform an asynchronous pattern search to minimize a function.

    A pattern of trial points is created around the current best point.
    No derivatives are calculated.  Instead, this pattern shrinks as the
    algorithm converges.  Tasks and results are submitted and collected
    fully asynchronously, and the current best point is updated as soon
    as possible.  This algorithm should be able to scale to use any
    number of cores, although there are practical limitations such as
    scheduler overhead and memory usage.  For example, using 100,000
    cores to minimize a 100-D objective function should work just fine.

    Parameters
    ----------
    client : dask.distributed.Client
        A client to a ``dask.distributed`` scheduler.
    func : callable
        The objective function to be minimized.  Must be in the form
        ``func(x, *args)`` where ``x`` is a 1-D array and ``args`` is
        a tuple of extra arguments passed to the objective function.
    x0 : ndarray
        1-D array; the initial guess.
    stepsize : ndarray
        1-D array; the initial step sizes for each dimension.  This may
        be repeatedly halved or doubled as the algorithm proceeds.  It
        is best to choose step sizes larger than the features you want
        the algorithm to step over.
    args : tuple, optional
        Extra arguments passed to the objective function.
    max_queue_size : int or None, optional
        Maximum number of active tasks to have submitted to the client.
        Default is the total number of threads plus the total number
        of worker processes of the the client's cluster.  This default
        is chosen to maximize occupancy of available cores, but, in
        general, ``max_queue_size`` does not need to be related to
        compute resources at all.  Choosing a larger ``max_queue_size``
        is the best way to improve robustness of the algorithm.
    min_queue_size : int or None, optional
        Minimum number of active tasks to have submitted to the client.
        Default is ``max_queue_size // 2``.
    min_new_submit : int, optional
        The minimum number of trial points to submit after a new best
        point has been found before accepting an even better point.
        This may help when there are multiple minima being explored.
    randomize : bool, optional
        Whether to randomize the order of trial points (default True).
    max_stencil_size: int or None, optional
        The maximum size of the stencil used to create the pattern of
        trial points around the current best point.  Default unlimited.
    stopratio : float, optional
        Termination condition: stop after the step size has been reduced
        by this amount.  Must be between 0 and 1.  Default is 0.01.
    max_tasks : int or None, optional
        Termination condition: stop after this many tasks have been
        completed.  Default unlimited.
    max_time : float or None, optional
        Termination condition: stop submitting new tasks after this many
        seconds have passed.  Default unlimited.
    integer_dimensions : array-like or None, optional
        1-D array; specify the indices of integer dimensions.

    Returns
    -------
    best_point: Point
        The optimization result.  ``best_point.point`` is the ndarray.
    results: dict
        All evalulated points and their scores.

    """
    # bound=None, low_memory_stencil=False
    if max_queue_size is None:
        ncores = client.ncores()
        max_queue_size = sum(ncores.values()) + len(ncores)
    if min_queue_size is None:
        min_queue_size = max(1, max_queue_size // 2)
    if max_stencil_size is None:
        max_stencil_size = 1e9
    x0 = np.array(x0)
    stepsize = np.array(stepsize)
    dims = len(stepsize)
    max_halvings = math.frexp(1 / stopratio)[1]
    stencil = RightHandedSimplexStencil(dims, max_halvings)
    gridsize = stepsize / 2.**max_halvings

    if integer_dimensions is not None:
        integer_dimensions = np.array(integer_dimensions)
        int_dims = np.zeros(len(x0), dtype=np.bool)
        int_dims[integer_dimensions] = 1
        x0[int_dims] = np.round(x0[int_dims])

    def to_grid(x):
        return np.round(x / gridsize) * gridsize

    orientation = np.ones(dims)
    cur_point = Point(to_grid(x0), -1)
    cur_point.start_time = time()
    cur_point.parent = cur_point
    cur_cost = result = np.inf
    is_contraction = True
    new_point = None

    if max_time is not None:
        end_time = time() + max_time
    results = {}
    running = {}
    processing = []
    contract_conditions = set()
    next_point = None
    next_cost = None

    # Begin from initial point
    future = client.submit(func, cur_point.point, *args)
    as_completed = distributed.as_completed([future], with_results=True)
    running[future] = cur_point
    results[cur_point] = None

    is_finished = False
    while not is_finished or running or next_point is not None or new_point is not None:
        if max_time is not None and time() > end_time:
            is_finished = True

        # Initialize new point
        if new_point is not None or is_contraction:
            if is_contraction:
                is_contraction = False
                if cur_point.stop_time is None:
                    cur_point.stop_time = time()
                new_point = Point(cur_point.point, cur_point.halvings + 1)
                new_point.parent = cur_point
                new_point.is_accepted = True
                new_point.result = cur_cost
                new_cost = cur_cost
                new_point.start_time = time()
            cur_point = new_point
            cur_cost = new_cost
            new_point = None
            new_cost = None
            cur_stepsize = to_grid(orientation * stepsize / 2.**cur_point.halvings)
            cur_added = 0
            contract_conditions.clear()
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
                len(running) < max_queue_size
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
                dx = step.point * cur_stepsize
                if integer_dimensions is not None:
                    # Round integer steps to the nearest integer away from zero
                    dx_ints = dx[int_dims]
                    dx[int_dims] = np.copysign(np.ceil(np.abs(dx_ints)), dx_ints)
                    trial_point = to_grid(cur_point.point + dx)
                    trial_point[int_dims] = np.round(trial_point[int_dims])
                    # Don't reduce the stepsize via the stencil if step is only ints
                    if step.halvings > 0 and (dx[~int_dims] != 0).sum() == 0:
                        halvings = cur_point.halvings
                else:
                    trial_point = to_grid(cur_point.point + dx)
                if halvings > max_halvings:
                    continue
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
                    future = client.submit(func, trial_point.point, *args)
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
        if running:
            block = (
                len(running) >= max_queue_size
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
        if next_point is not None and (cur_added >= min_new_submit or stencil_index >= max_stencil_size or is_finished):
            results[next_point] = next_cost
            next_point.result = next_cost
            contract_conditions.discard(next_point)
            if next_cost < cur_cost:
                next_point.is_accepted = True
                # Orient the asymmetric stencil towards the expected direction
                # of descent based on the old and new points and their parents.
                # The stencil prefers negative directions, so this is the correct sign.
                diff = (
                    (next_point.point - next_point.parent.point)
                    + (cur_point.point - cur_point.parent.point)
                )
                orientation = np.where(diff, np.copysign(orientation, diff), orientation)
                new_point = next_point
                new_cost = next_cost

            for point, result in processing:
                results[point] = result
                point.result = result
                contract_conditions.discard(point)
            if new_point is None and not contract_conditions and stencil_index > 2 * dims:
                is_contraction = True
            next_point = None
            next_cost = None
            processing[:] = []
        elif next_point is None and stencil_index >= max_stencil_size and not running and not is_finished:
            # Nothing running, nothing to process, and nothing to submit, so contract
            is_contraction = True

    return cur_point, results

