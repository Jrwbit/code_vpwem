"""Bounded directed roadmap for offline fixed-waist adjustment connections.

It biases samples around the two known valid arm configurations. Every returned
edge is independently rechecked with the caller's full-density validator.
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np


def _shortest(graph, blocked):
    distance = {0: 0.0}
    parent = {}
    queue = [(0.0, 0)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distance.get(node):
            continue
        if node == 1:
            path = [1]
            while path[-1] != 0:
                path.append(parent[path[-1]])
            return path[::-1]
        for nxt, weight in graph[node]:
            if tuple(sorted((node, nxt))) in blocked:
                continue
            total = cost + weight
            if total < distance.get(nxt, float('inf')):
                distance[nxt] = total
                parent[nxt] = node
                heapq.heappush(queue, (total, nxt))
    return None


def directed_connect(ctx, start, end, rng, validator, *, seconds=12.0,
                     n_samples=64, coarse_deg=2.0, seed_arm_rows=()):
    """Return ``{ok, rows, checks, ...}``; a failed result is never a route.

    Inputs are the four fixed model waist angles followed by seven arm angles.
    The existing adjustment exclusion zones remain active (phase ``transit``).
    A coarse edge is only a search hint; returned edges use the default policy.
    """
    began = time.monotonic()
    deadline = began + seconds
    A, B = np.asarray(start, float), np.asarray(end, float)
    if A.shape != (11,) or B.shape != (11,) or not np.allclose(A[:4], B[:4], atol=1e-12, rtol=0):
        raise ValueError('Adjustment connector requires two 11-axis rows with one identical waist')
    fid = ctx.fixture
    waist = A[:4]
    full = validator.ClearancePolicy()
    coarse = validator.ClearancePolicy(sample_deg=coarse_deg)
    stats = {'point_checks': 0, 'coarse_edge_checks': 0, 'strict_edge_checks': 0,
             'strict_rejected_edges': 0, 'sampling_mode': 'local interpolation and endpoint Gaussian biases'}
    def check(a, b, policy):
        return validator.check_segment(ctx, a, b, 'transit', fid, policy, waist=waist)
    def finish(ok, **extra):
        return {'ok': ok, 'elapsed_s': time.monotonic() - began, **stats, **extra}
    for label, q in [('start', A), ('end', B)]:
        result = check(q, q, full); stats['point_checks'] += 1
        if not result['ok']:
            return finish(False, reason=label + '_is_not_a_legal_adjustment_pose', endpoint_check=result)
    direct = check(A, B, full); stats['strict_edge_checks'] += 1
    if direct['ok']:
        return finish(True, rows=[A, B], checks=[direct])
    from sim.helios_insert import scene
    lo = np.array([ctx.robot.lim[n][0] for n in scene.LEFT]) + math.radians(3)
    hi = np.array([ctx.robot.lim[n][1] for n in scene.LEFT]) - math.radians(3)
    points = [A, B]
    attempts = 0
    seeds = list(seed_arm_rows)
    while len(points) < n_samples + 2 and time.monotonic() < deadline:
        attempts += 1
        if attempts > n_samples * 8:
            break
        if seeds:
            arm = np.asarray(seeds.pop(0), float)
            if arm.shape == (11,):
                arm = arm[4:]
        else:
            mode = attempts % 10
            alpha = rng.uniform(.05, .95)
            center = A[4:] * (1 - alpha) + B[4:] * alpha
            if mode == 0:
                arm = rng.uniform(lo, hi)
            elif mode in (1, 2):
                arm = A[4:] + rng.normal(0, .35, 7)
            elif mode in (3, 4):
                arm = B[4:] + rng.normal(0, .35, 7)
            else:
                arm = center + rng.normal(0, .35 if mode < 8 else .7, 7)
        q = np.r_[waist, np.clip(arm, lo, hi)]
        if any(np.max(np.abs(q - old)) < 1e-6 for old in points):
            continue
        result = check(q, q, full); stats['point_checks'] += 1
        if result['ok']:
            points.append(q)
    stats['accepted_roadmap_points'] = len(points)
    vertices = np.asarray(points)
    distances = np.linalg.norm(vertices[:, None, 4:] - vertices[None, :, 4:], axis=2)
    graph = {i: {} for i in range(len(points))}
    for i in range(len(points)):
        # Endpoints connect to every candidate. Interior nearest-neighbour
        # edges allow two or more small corrections without a quadratic scan.
        near = range(len(points)) if i < 2 else np.argsort(distances[i])[1:11]
        for j in near:
            if i != j:
                graph[i][int(j)] = graph[int(j)][i] = float(distances[i, j])
    graph = {i: list(edges.items()) for i, edges in graph.items()}
    blocked = {(0, 1)}
    checked = {}
    last_failure = direct
    while time.monotonic() < deadline:
        route = _shortest(graph, blocked)
        if route is None:
            return finish(False, reason='directed_roadmap_disconnected', last_failure=last_failure)
        edges = list(zip(route, route[1:]))
        invalid = False
        for i, j in edges:
            key = tuple(sorted((i, j)))
            if key in checked:
                continue
            result = check(points[i], points[j], coarse); stats['coarse_edge_checks'] += 1
            if not result['ok']:
                blocked.add(key); last_failure = result; invalid = True; break
            checked[key] = None
        if invalid:
            continue
        checks = []
        for i, j in edges:
            key = tuple(sorted((i, j)))
            result = checked[key]
            if result is None:
                result = check(points[i], points[j], full); stats['strict_edge_checks'] += 1
                checked[key] = result
            if not result['ok']:
                blocked.add(key); last_failure = result; stats['strict_rejected_edges'] += 1; invalid = True; break
            checks.append(result)
        if not invalid:
            return finish(True, rows=[points[i] for i in route], checks=checks)
    return finish(False, reason='directed_connection_time_budget', last_failure=last_failure)
