from __future__ import absolute_import, division, print_function

import numpy as np
import time
from distributed import Client
from distributed.utils_test import cluster, loop

from dask_patternsearch import search


def sphere(x):
    return x.dot(x)


def test_convergence_2d_simple(loop):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as client:
            x0 = np.array([10., 15])
            stepsize = np.array([1., 1])
            stopratio = 1e-2
            best, results = search(client, sphere, x0, stepsize, stopratio=stopratio)
            assert (np.abs(best.point) < 2*stopratio).all()

            best, results = search(client, sphere, x0, stepsize, stopratio=stopratio, max_queue_size=20)
            assert (np.abs(best.point) < 2*stopratio).all()

            best, results = search(client, sphere, x0, stepsize, stopratio=stopratio, max_queue_size=1)
            assert (np.abs(best.point) < 2*stopratio).all()

            best, results = search(client, sphere, x0, stepsize, stopratio=stopratio, min_new_submit=4)
            assert (np.abs(best.point) < 2*stopratio).all()

            best, results = search(client, sphere, x0, stepsize, stopratio=stopratio, max_tasks=10)
            assert len(results) == 10

            best, results = search(client, sphere, x0, stepsize, stopratio=stopratio, max_stencil_size=4)
            assert (np.abs(best.point) < 2*stopratio).all()

            best, results = search(client, sphere, x0, stepsize, stopratio=stopratio, max_stencil_size=4, min_new_submit=4)
            assert (np.abs(best.point) < 2*stopratio).all()

