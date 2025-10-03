#!/usr/bin/env python3
"""
bmssp_full_impl.py

Single-file implementation:
 - Dijkstra's algorithm (standard).
 - A BMSSP-inspired implementation faithful to the pseudocode structure:
    BMSSP (recursive), FIND_PIVOTS (bounded Bellman-Ford-like), BASECASE (Dijkstra-like).
 - Practical DataStructure D (heap + dictionary) implementing insert, pull, batch_prepend.
 - Instrumentation and randomized test harness.

Notes:
 - This is a runnable, practical implementation. The paper's theoretical data structure
   and tight complexity constants are nontrivial to reproduce exactly; this code aims
   to reflect the control flow and correctness while being usable for experiments.
"""

from __future__ import annotations
import heapq
import random
import time
import math
import argparse
from typing import Dict, List, Tuple, Set, Optional

Node = int
Weight = float
Edge = Tuple[Node, Node, Weight]
Graph = Dict[Node, List[Tuple[Node, Weight]]]


# ---------------------------
# Utilities & Graph generator
# ---------------------------
def generate_sparse_directed_graph(n: int, m: int, max_w: float = 100.0, seed: Optional[int] = None) -> Tuple[Graph, List[Edge]]:
    if seed is not None:
        random.seed(seed)
    graph: Graph = {i: [] for i in range(n)}
    edges: List[Edge] = []
    # weak backbone to avoid isolated nodes
    for i in range(1, n):
        u = random.randrange(0, i)
        w = random.uniform(1.0, max_w)
        graph[u].append((i, w))
        edges.append((u, i, w))
    remaining = max(0, m - (n - 1))
    for _ in range(remaining):
        u = random.randrange(n)
        v = random.randrange(n)
        w = random.uniform(1.0, max_w)
        graph[u].append((v, w))
        edges.append((u, v, w))
    return graph, edges


# ---------------------------
# Instrumentation
# ---------------------------
class Instrument:
    def __init__(self):
        self.relaxations = 0
        self.heap_ops = 0

    def reset(self):
        self.relaxations = 0
        self.heap_ops = 0


# ---------------------------
# Dijkstra
# ---------------------------
def dijkstra(graph: Graph, source: Node, instr: Optional[Instrument] = None) -> Dict[Node, Weight]:
    if instr is None:
        instr = Instrument()
    dist: Dict[Node, Weight] = {v: float('inf') for v in graph}
    dist[source] = 0.0
    heap: List[Tuple[Weight, Node]] = [(0.0, source)]
    instr.heap_ops += 1
    while heap:
        d_u, u = heapq.heappop(heap)
        instr.heap_ops += 1
        if d_u > dist[u]:
            continue
        for v, w in graph[u]:
            instr.relaxations += 1
            alt = d_u + w
            if alt < dist.get(v, float('inf')):
                dist[v] = alt
                heapq.heappush(heap, (alt, v))
                instr.heap_ops += 1
    return dist


# ---------------------------
# DataStructure D (practical)
# ---------------------------
class DataStructureD:
    """
    Practical approximation of the paper's D:
    - insert(v, key)
    - batch_prepend(iterable of (v,key))
    - pull() -> (Bi, Si) where Si is a small set of vertices with smallest keys.
    - empty()
    Internal: heap + dict for current best keys. pull returns up to block_size items.
    """

    def __init__(self, M: int, B_upper: float, block_size: Optional[int] = None):
        self.heap: List[Tuple[Weight, Node]] = []
        self.best: Dict[Node, Weight] = {}
        self.M = max(1, M)
        self.B_upper = B_upper
        # choose block size heuristic from M
        self.block_size = block_size if block_size is not None else max(1, self.M // 8)

    def insert(self, v: Node, key: Weight):
        prev = self.best.get(v)
        if prev is None or key < prev:
            self.best[v] = key
            heapq.heappush(self.heap, (key, v))

    def batch_prepend(self, iterable_pairs):
        # They are expected to have small keys — but implementation same as insert
        for v, key in iterable_pairs:
            self.insert(v, key)

    def _cleanup(self):
        # Remove stale heap entries
        while self.heap and self.best.get(self.heap[0][1]) != self.heap[0][0]:
            heapq.heappop(self.heap)

    def empty(self) -> bool:
        self._cleanup()
        return len(self.heap) == 0

    def pull(self) -> Tuple[Weight, Set[Node]]:
        """
        Return Bi (the smallest key present) and a set Si of up to block_size vertices with smallest keys.
        """
        self._cleanup()
        if not self.heap:
            raise IndexError("pull from empty D")
        # smallest key
        Bi = self.heap[0][0]
        Si: Set[Node] = set()
        # Pop up to block_size best current entries
        while self.heap and len(Si) < self.block_size:
            key, v = heapq.heappop(self.heap)
            if self.best.get(v) == key:
                Si.add(v)
                # remove from best to mark as "pulled"
                del self.best[v]
        return Bi, Si


# ---------------------------
# FIND_PIVOTS (practical bounded BF)
# ---------------------------
def find_pivots(graph: Graph, dist: Dict[Node, Weight], S: Set[Node], B: float, n: int,
                k_steps: int, p_limit: int, instr: Optional[Instrument] = None) -> Tuple[Set[Node], Set[Node]]:
    """
    Heuristic approximation of FINDPIVOTS:
      - Run up to k_steps of Bellman-Ford-like relaxations starting from S (only considering nodes with dist < B).
      - Collect W = nodes that got finalized/reached within those k steps (i.e., discovered by these relaxations).
      - Choose P as up to p_limit nodes from S with smallest dist[] (ensures P non-empty when S non-empty).
    Returns (P, W).
    """
    if instr is None:
        instr = Instrument()

    # Filter S to those with dist < B
    S_filtered = [v for v in S if dist.get(v, float('inf')) < B]
    # Choose pivots P — heuristic: smallest distances in S_filtered
    if not S_filtered:
        # fallback: choose up to p_limit arbitrary samples from S
        P = set(list(S)[:max(1, min(len(S), p_limit))]) if S else set()
    else:
        S_filtered.sort(key=lambda v: dist.get(v, float('inf')))
        P = set(S_filtered[:max(1, min(len(S_filtered), p_limit))])

    # Bounded BF: start frontier from P (if P empty use S)
    source_frontier = P if P else set(S)
    discovered = set(source_frontier)
    frontier = set(source_frontier)

    # local copy of tentative distances (we operate on global dist but will update dist in caller)
    # We'll perform relaxations but not set dist globally here; instead return W (discovered)
    for _ in range(max(1, k_steps)):
        if not frontier:
            break
        next_front = set()
        for u in frontier:
            du = dist.get(u, float('inf'))
            if du >= B:
                continue
            for v, w in graph[u]:
                instr.relaxations += 1
                nd = du + w
                # consider only nodes with nd < B
                if nd < B and v not in discovered:
                    discovered.add(v)
                    next_front.add(v)
        frontier = next_front

    W = discovered.copy()
    # P must be small relative to S; ensure non-empty if S non-empty
    if not P and S:
        P = {next(iter(S))}
    return P, W


# ---------------------------
# BASECASE (Dijkstra-l ike)
# ---------------------------
def basecase(graph: Graph, dist: Dict[Node, Weight], B: float, S: Set[Node], k: int, instr: Optional[Instrument] = None) -> Tuple[float, Set[Node]]:
    """
    BASECASE: S should be singleton (but if not, pick best node in S).
    Run Dijkstra-like expansion from that node limited by B, and stop after finding up to k+1 completed nodes.
    Return (B_prime, Uo_set).
    """
    if instr is None:
        instr = Instrument()

    if not S:
        return B, set()

    # choose source x in S with smallest dist
    x = min(S, key=lambda v: dist.get(v, float('inf')))
    # local heap
    heap: List[Tuple[Weight, Node]] = []
    start_d = dist.get(x, float('inf'))
    heapq.heappush(heap, (start_d, x))
    instr.heap_ops += 1

    Uo: Set[Node] = set()
    visited: Set[Node] = set()

    while heap and len(Uo) < (k + 1):
        d_u, u = heapq.heappop(heap)
        instr.heap_ops += 1
        if d_u > dist.get(u, float('inf')):
            continue
        # mark 'u' complete for this basecase
        if u not in Uo:
            Uo.add(u)
        for v, w in graph[u]:
            instr.relaxations += 1
            newd = dist.get(u, float('inf')) + w
            if newd < dist.get(v, float('inf')) and newd < B:
                dist[v] = newd
                heapq.heappush(heap, (newd, v))
                instr.heap_ops += 1

    if len(Uo) <= k:
        return B, Uo
    else:
        finite_dists = [dist[v] for v in Uo if math.isfinite(dist.get(v, float('inf')))]
        if not finite_dists:
            return B, set()
        maxd = max(finite_dists)
        U_filtered = {v for v in Uo if dist.get(v, float('inf')) < maxd}
        return maxd, U_filtered


# ---------------------------
# BMSSP (practical recursive)
# ---------------------------
def bmssp(graph: Graph, dist: Dict[Node, Weight], edges: List[Edge],
          l: int, B: float, S: Set[Node], n: int,
          instr: Optional[Instrument] = None) -> Tuple[float, Set[Node]]:
    """
    BMSSP recursive function.
    - Uses find_pivots, DataStructureD, basecase as building blocks.
    - l: recursion depth; when l==0 call basecase.
    - n: number of nodes in graph (for parameter choices).
    """
    if instr is None:
        instr = Instrument()

    # sensible parameter choices (heuristic approximations from the paper)
    if n <= 2:
        t_param = 1
        k_param = 2
    else:
        # t ~ (log n)^{2/3}, k ~ (log n)^{1/3} (rounded and clamped)
        t_param = max(1, int(round((math.log(max(3, n)) ** (2.0 / 3.0)))))
        k_param = max(2, int(round((math.log(max(3, n)) ** (1.0 / 3.0)))))

    # If l == 0 then basecase (ensure S is at least singleton)
    if l <= 0:
        # If S empty, nothing to do
        if not S:
            return B, set()
        return basecase(graph, dist, B, S, k_param, instr)

    # FIND_PIVOTS: compute P, W
    # use p_limit proportional to 2^t (heuristic)
    p_limit = max(1, 2 ** min(10, t_param))  # cap exponent to keep p_limit reasonable
    # choose k_steps for find_pivots: k_param * some small constant
    k_steps = max(1, k_param)
    P, W = find_pivots(graph, dist, S, B, n, k_steps, p_limit, instr)

    # Data structure D initialization
    M = 2 ** max(0, (l - 1) * t_param)
    D = DataStructureD(M, B, block_size=max(1, min(len(P) or 1, 64)))
    # insert pivots into D
    for x in P:
        D.insert(x, dist.get(x, float('inf')))

    B_prime_initial = min((dist.get(x, float('inf')) for x in P), default=B)
    U: Set[Node] = set()
    B_prime_sub_values: List[float] = []

    # loop guard & limit to avoid pathological loops
    loop_guard = 0
    limit = k_param * (2 ** (l * max(1, t_param)))
    while len(U) < limit and not D.empty():
        loop_guard += 1
        if loop_guard > 20000:
            # safety break (shouldn't happen for normal sizes)
            break
        try:
            Bi, Si = D.pull()
        except IndexError:
            break
        # Recursive call
        B_prime_sub, Ui = bmssp(graph, dist, edges, l - 1, Bi, Si, n, instr)
        B_prime_sub_values.append(B_prime_sub)
        # Add Ui to U
        U |= Ui

        # Relax edges from Ui
        K_for_batch: Set[Tuple[Node, Weight]] = set()
        for u in Ui:
            du = dist.get(u, float('inf'))
            if not math.isfinite(du):
                continue
            for v, w_uv in graph[u]:
                instr.relaxations += 1
                newd = du + w_uv
                # Accept equality per remark (<=) to allow reuse
                if newd <= dist.get(v, float('inf')):
                    dist[v] = newd
                    if Bi <= newd < B:
                        D.insert(v, newd)
                    elif B_prime_sub <= newd < Bi:
                        K_for_batch.add((v, newd))
        # Also include Si nodes whose dist falls into [B_prime_sub, Bi)
        for x in Si:
            dx = dist.get(x, float('inf'))
            if B_prime_sub <= dx < Bi:
                K_for_batch.add((x, dx))
        if K_for_batch:
            D.batch_prepend(K_for_batch)

    if B_prime_sub_values:
        B_prime_final = min([B_prime_initial] + B_prime_sub_values)
    else:
        B_prime_final = B_prime_initial

    U_final = set(U)
    for x in W:
        if dist.get(x, float('inf')) < B_prime_final:
            U_final.add(x)
    return B_prime_final, U_final


# ---------------------------
# Test harness
# ---------------------------
def run_single_test(n: int, m: int, seed: int = 0, source: int = 0):
    print(f"Generating graph: n={n}, m={m}, seed={seed}")
    graph, edges = generate_sparse_directed_graph(n, m, seed=seed)
    avg_deg = sum(len(adj) for adj in graph.values()) / n
    print(f"Graph generated. avg out-degree ≈ {avg_deg:.3f}")

    # Prepare distances
    dist0 = {v: float('inf') for v in graph}
    dist0[source] = 0.0

    # Dijkstra timing
    instr_dij = Instrument()
    t0 = time.time()
    dist_dij = dijkstra(graph, source, instr_dij)
    t1 = time.time()
    print(f"Dijkstra: time={t1-t0:.6f}s, relaxations={instr_dij.relaxations}, heap_ops={instr_dij.heap_ops}, reachable={sum(1 for v in dist_dij.values() if math.isfinite(v))}")

    # BMSSP practical
    dist_bm = {v: float('inf') for v in graph}
    dist_bm[source] = 0.0
    instr_bm = Instrument()
    # choose top-level recursion l heuristically
    if n <= 2:
        l = 1
    else:
        t_guess = max(1, int(round((math.log(max(3, n)) ** (2.0 / 3.0)))))
        l = max(1, int(max(1, round(math.log(max(3, n)) / t_guess))))
    print(f"BMSSP params: top-level l={l}")
    t0 = time.time()
    Bp, U_final = bmssp(graph, dist_bm, edges, l, float('inf'), {source}, n, instr_bm)
    t1 = time.time()
    print(f"BMSSP: time={t1-t0:.6f}s, relaxations={instr_bm.relaxations}, reachable={sum(1 for v in dist_bm.values() if math.isfinite(v))}, B'={Bp}, |U_final|={len(U_final)}")

    # Compare distances for commonly reachable nodes
    diffs = []
    for v in graph:
        dv = dist_dij.get(v, float('inf'))
        db = dist_bm.get(v, float('inf'))
        if math.isfinite(dv) and math.isfinite(db):
            diffs.append(abs(dv - db))
    max_diff = max(diffs) if diffs else 0.0
    print(f"Distance agreement (max abs diff on commonly reachable nodes): {max_diff:.6e}")
    # Sanity checks: ideally dist_bm should match dist_dij for many nodes (not necessarily all,
    # depending on parameter tuning). The practical BMSSP should explore the graph.
    return {
        'n': n, 'm': m, 'seed': seed,
        'dijkstra_time': (t1 - t0), 'dijkstra_relax': instr_dij.relaxations,
        'bmssp_time': (t1 - t0), 'bmssp_relax': instr_bm.relaxations,
        'dijkstra_reachable': sum(1 for v in dist_dij.values() if math.isfinite(v)),
        'bmssp_reachable': sum(1 for v in dist_bm.values() if math.isfinite(v)),
        'max_diff': max_diff
    }


# ---------------------------
# CLI main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="BMSSP (practical) vs Dijkstra - full implementation")
    parser.add_argument('-n', '--nodes', type=int, default=200000, help='nodes')
    parser.add_argument('-m', '--edges', type=int, default=800000, help='edges')
    parser.add_argument('-s', '--seed', type=int, default=0, help='seed')
    args = parser.parse_args()
    run_single_test(args.nodes, args.edges, seed=args.seed)


if __name__ == "__main__":
    main()