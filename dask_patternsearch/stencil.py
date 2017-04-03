from __future__ import absolute_import, division, print_function

import numpy as np
from toolz import concatv, drop, interleave


class SimplexPoint(object):
    __slots__ = ('stencil', 'stepsize', 'halvings', 'index', 'is_reflect',
                 'is_doubled', 'simplex_key', 'point_key', 'point')

    def __init__(self, point, parent, index, is_reflect=False, is_contract=False):
        self.stencil = parent.stencil
        self.stepsize = parent.stepsize
        self.halvings = parent.halvings
        self.index = index
        self.is_reflect = is_reflect
        self.is_doubled = is_reflect and parent.is_reflect and index == 1 and not parent.is_doubled
        if is_contract:
            self.stepsize = self.stencil.to_grid(0.5 * self.stepsize)
            self.halvings += 1
        elif self.is_doubled:
            self.stepsize = self.stencil.to_grid(2 * self.stepsize)
            self.halvings -= 1
        self.simplex_key = self.stencil.get_simplex_key(parent, index, is_reflect)
        self.point_key = self.stencil.get_point_key(point)
        self.point = self.stencil.get_point(self.point_key)

    @property
    def simplex(self):
        return self.stencil.get_simplex(self.simplex_key)

    def get_points(self):
        points = self.stencil.to_grid(self.point + self.stepsize * self.simplex)
        return (SimplexPoint(x, self, i) for i, x in drop(1, enumerate(points)))

    def get_reflections(self):
        if self.index == 0 and self.is_reflect and not self.is_doubled:
            return iter([])
        points = self.stencil.to_grid(self.point - self.stepsize * self.simplex)
        return (SimplexPoint(x, self, i, is_reflect=True) for i, x in enumerate(points))

    def get_contractions(self):
        if self.halvings > self.stencil.max_halvings:
            return iter([])
        points = self.stencil.to_grid(self.point + 0.5 * self.stepsize * self.simplex)
        return (SimplexPoint(x, self, i, is_contract=True) for i, x in enumerate(points))

    def __hash__(self):
        return hash((self.point_key, self.simplex_key, self.index, self.halvings,
                     self.is_reflect, self.is_doubled))

    def __eq__(self, other):
        return (
            self.point_key == other.point_key
            and self.simplex_key == other.simplex_key
            and self.index == other.index
            and self.halvings == other.halvings
            and self.is_reflect == other.is_reflect
            and self.is_doubled == other.is_doubled
            and self.stencil is other.stencil
        )

    def __repr__(self):
        return type(self).__name__ + repr(self.point)[len('array'):]


class InitialSimplexPoint(object):
    def __init__(self, stencil):
        self.stencil = stencil
        self.simplex = stencil.simplex
        self.halvings = 0
        self.stepsize = 1.0
        self.is_reflect = False
        self.is_doubled = False


class RightHandedSimplexStencil(object):
    def __init__(self, dims, max_halvings):
        self.dims = dims
        self.simplex_intern = {}
        self.point_intern = {}
        self.point_cache = {}
        self.max_halvings = max_halvings
        self.gridsize = 2.**(-max_halvings-1)
        r = np.arange(dims + 1)
        self.indexers = np.stack([
            np.concatenate([[i], r[:i], r[i+1:]])
            for i in range(dims + 1)
        ])
        self.simplex = np.concatenate(
            [np.zeros((1, dims), dtype=np.int8), np.identity(dims, dtype=np.int8)],
            axis=0
        )
        self.point = np.zeros(dims)
        self.get_simplex_key(self, 0, False)
        self.get_point_key(self.point)
        self._stencil_points = []  # This serves as a cache for generated stencil points
        self._stencil_iter = self._generate_stencil()

    def get_simplex_key(self, parent, index, is_reflect):
        simplex = parent.simplex
        if index != 0:
            simplex = (simplex - simplex[index])[self.indexers[index]]
        if is_reflect:
            simplex = -simplex
        key = simplex.tostring()
        if key in self.simplex_intern:
            return self.simplex_intern[key]
        self.simplex_intern[key] = key
        return key

    def get_simplex(self, key):
        return np.fromstring(key, np.int8).reshape((self.dims + 1, self.dims))

    def get_point_key(self, point):
        key = point.tostring()
        if key in self.point_cache:
            return self.point_intern[key]
        self.point_intern[key] = key
        self.point_cache[key] = point
        return key

    def get_point(self, key):
        return self.point_cache[key]

    def to_grid(self, x):
        return np.round(x / self.gridsize) * self.gridsize

    def _generate_stencil(self):
        init = InitialSimplexPoint(self)
        point = SimplexPoint(self.point, init, 0)
        seen = {point}
        seen_add = seen.add
        first_seen = {point.point_key}
        first_seen_add = first_seen.add
        stencil_points_append = self._stencil_points.append

        for p in point.get_points():
            stencil_points_append(p)
            yield p
            first_seen_add(p.point_key)
            seen_add(p)

        self_reflect = []
        mirror_reflect = []
        reflect = []
        self_contract = [point]
        contract = []

        while True:
            next_self_reflect = []
            next_mirror_reflect = []
            next_reflect = []
            next_self_contract = []
            next_contract = []
            for p in concatv(
                interleave(x.get_reflections() for x in self_reflect),
                interleave(x.get_reflections() for x in mirror_reflect),
                interleave(x.get_reflections() for x in reflect),
                interleave(x.get_reflections() for x in self_contract),
                interleave(x.get_reflections() for x in contract),
            ):
                if p.point_key not in first_seen:
                    stencil_points_append(p)
                    yield p
                    first_seen_add(p.point_key)
                    seen_add(p)
                    next_reflect.append(p)
                elif p not in seen:
                    seen_add(p)
                    if p.index == 0:
                        next_self_reflect.append(p)
                    elif p.index == 1:
                        next_mirror_reflect.append(p)
                    else:
                        next_reflect.append(p)
            for p in concatv(
                interleave(x.get_contractions() for x in self_reflect),
                interleave(x.get_contractions() for x in mirror_reflect),
                interleave(x.get_contractions() for x in reflect),
                interleave(x.get_contractions() for x in self_contract),
                interleave(x.get_contractions() for x in contract),
            ):
                if p.point_key not in first_seen:
                    stencil_points_append(p)
                    yield p
                    first_seen_add(p.point_key)
                    seen_add(p)
                    next_contract.append(p)
                elif p not in seen:
                    seen_add(p)
                    if p.index == 0:
                        next_self_contract.append(p)
                    else:
                        next_contract.append(p)
            self_reflect = next_self_reflect
            mirror_reflect = next_mirror_reflect
            reflect = next_reflect
            self_contract = next_self_contract
            contract = next_contract

    def generate_stencil_points(self):
        return concatv(self._stencil_points, self._stencil_iter)

