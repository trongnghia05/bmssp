"""

Triển khai đầy đủ trong một file:
 - Thuật toán Dijkstra (chuẩn).
 - Một số chỗ khác BMSSP papper, do không chạy được, bám sát cấu trúc giả mã:
    BMSSP (đệ quy), FIND_PIVOTS (giống Bellman-Ford có giới hạn), BASECASE (tương tự Dijkstra).
 - Cấu trúc dữ liệu thực tế D (heap + dictionary) cài đặt insert, pull, batch_prepend.
 - Các phép đo hiệu năng và bộ test ngẫu nhiên.

Ghi chú:
 - Đây là một bản cài đặt thực tế có thể chạy được. Cấu trúc dữ liệu lý thuyết và hằng số độ phức tạp chặt trong bài báo rất khó implement chính xác; mã này hướng đến việc phản ánh đúng luồng điều khiển và tính đúng đắn, đồng thời có thể dùng để thử nghiệm.
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
# Bộ sinh đồ thị
# ---------------------------
def generate_sparse_directed_graph(n: int, m: int, max_w: float = 100.0, seed: Optional[int] = None) -> Tuple[Graph, List[Edge]]:
    if seed is not None:
        random.seed(seed)
    graph: Graph = {i: [] for i in range(n)}
    edges: List[Edge] = []
    # Xây dựng "xương sống" yếu để tránh nút cô lập
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
# Đánh dấu đỉnh
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
# Cấu trúc dữ liệu D
# ---------------------------
class DataStructureD:
    """
    Phiên bản xấp xỉ thực tế của D trong bài báo:
    - insert(v, key)
    - batch_prepend(danh sách (v, key))
    - pull() -> (Bi, Si), trong đó Si là tập nhỏ các đỉnh có khóa nhỏ nhất.
    - empty()
    Nội bộ: heap + dict lưu khóa hiện tại tốt nhất. pull trả về tối đa block_size phần tử.
    """

    def __init__(self, M: int, B_upper: float, block_size: Optional[int] = None):
        self.heap: List[Tuple[Weight, Node]] = []
        self.best: Dict[Node, Weight] = {}
        self.M = max(1, M)
        self.B_upper = B_upper
        # Chọn block_size dựa theo M (heuristic)
        self.block_size = block_size if block_size is not None else max(1, self.M // 8)

    def insert(self, v: Node, key: Weight):
        prev = self.best.get(v)
        if prev is None or key < prev:
            self.best[v] = key
            heapq.heappush(self.heap, (key, v))

    def batch_prepend(self, iterable_pairs):
        # Các phần tử này được kỳ vọng có key nhỏ — nhưng cài đặt giống insert
        for v, key in iterable_pairs:
            self.insert(v, key)

    def _cleanup(self):
        # Loại bỏ các phần tử cũ trong heap
        while self.heap and self.best.get(self.heap[0][1]) != self.heap[0][0]:
            heapq.heappop(self.heap)

    def empty(self) -> bool:
        self._cleanup()
        return len(self.heap) == 0

    def pull(self) -> Tuple[Weight, Set[Node]]:
        """
        Trả về Bi (khóa nhỏ nhất hiện có) và tập Si gồm tối đa block_size đỉnh có khoảng cách nhỏ nhất.
        """
        self._cleanup()
        if not self.heap:
            raise IndexError("pull từ D rỗng")
        # Khóa nhỏ nhất
        Bi = self.heap[0][0]
        Si: Set[Node] = set()
        # Lấy tối đa block_size phần tử tốt nhất
        while self.heap and len(Si) < self.block_size:
            key, v = heapq.heappop(self.heap)
            if self.best.get(v) == key:
                Si.add(v)
                # Xóa khỏi best để đánh dấu đã lấy ra
                del self.best[v]
        return Bi, Si


# ---------------------------
# FIND_PIVOTS (giống Bellman-Ford có giới hạn)
# ---------------------------
def find_pivots(graph: Graph, dist: Dict[Node, Weight], S: Set[Node], B: float, n: int,
                k_steps: int, p_limit: int, instr: Optional[Instrument] = None) -> Tuple[Set[Node], Set[Node]]:
    """
    Xấp xỉ heuristic của FIND_PIVOTS:
      - Chạy tối đa k_steps vòng lặp relax kiểu Bellman-Ford bắt đầu từ S (chỉ xét các nút có dist < B).
      - W = tập các node được phát hiện trong các bước này.
      - Chọn P gồm tối đa p_limit node (đỉnh) trong S có dist nhỏ nhất.
    Trả về (P, W).
    """
    if instr is None:
        instr = Instrument()

    # Lọc S chỉ còn các đỉnh có dist < B
    S_filtered = [v for v in S if dist.get(v, float('inf')) < B]
    # Chọn các pivot P — heuristic: dist nhỏ nhất trong S_filtered
    if not S_filtered:
        # Nếu trống, chọn ngẫu nhiên tối đa p_limit phần tử trong S
        P = set(list(S)[:max(1, min(len(S), p_limit))]) if S else set()
    else:
        S_filtered.sort(key=lambda v: dist.get(v, float('inf')))
        P = set(S_filtered[:max(1, min(len(S_filtered), p_limit))])

    # Bellman-Ford giới hạn: bắt đầu từ P (nếu P rỗng thì dùng S)
    source_frontier = P if P else set(S)
    discovered = set(source_frontier)
    frontier = set(source_frontier)

    # Bản sao cục bộ của dist; chỉ dùng tạm trong vòng lặp
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
                if nd < B and v not in discovered:
                    discovered.add(v)
                    next_front.add(v)
        frontier = next_front

    W = discovered.copy()
    if not P and S:
        P = {next(iter(S))}
    return P, W


# ---------------------------
# BASECASE (tương tự Dijkstra)
# ---------------------------
def basecase(graph: Graph, dist: Dict[Node, Weight], B: float, S: Set[Node], k: int, instr: Optional[Instrument] = None) -> Tuple[float, Set[Node]]:
    """
    BASECASE: S nên là tập đơn (nhưng nếu không, chọn nút có dist nhỏ nhất).
    Chạy mở rộng kiểu Dijkstra giới hạn bởi B, dừng sau khi tìm được tối đa k+1 nút hoàn tất.
    Trả về (B_prime, Uo_set).
    """
    if instr is None:
        instr = Instrument()

    if not S:
        return B, set()

    # Chọn đỉnh x trong S có dist nhỏ nhất
    x = min(S, key=lambda v: dist.get(v, float('inf')))
    # Heap cục bộ
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
# BMSSP (phiên bản đệ quy thực tế)
# ---------------------------
def bmssp(graph: Graph, dist: Dict[Node, Weight], edges: List[Edge],
          l: int, B: float, S: Set[Node], n: int,
          instr: Optional[Instrument] = None) -> Tuple[float, Set[Node]]:
    """
    Hàm đệ quy BMSSP.
    - Dùng find_pivots, DataStructureD, basecase làm các khối cơ bản.
    - l: độ sâu đệ quy; khi l==0 thì gọi basecase.
    - n: số đỉnh của đồ thị (dùng để chọn tham số).
    """
    if instr is None:
        instr = Instrument()

    # Chọn tham số hợp lý (xấp xỉ heuristic từ bài báo)
    if n <= 2:
        t_param = 1
        k_param = 2
    else:
        t_param = max(1, int(round((math.log(max(3, n)) ** (2.0 / 3.0)))))
        k_param = max(2, int(round((math.log(max(3, n)) ** (1.0 / 3.0)))))

    # Nếu l == 0 thì gọi basecase
    if l <= 0:
        if not S:
            return B, set()
        return basecase(graph, dist, B, S, k_param, instr)

    # Tìm P, W
    p_limit = max(1, 2 ** min(10, t_param))
    k_steps = max(1, k_param)
    P, W = find_pivots(graph, dist, S, B, n, k_steps, p_limit, instr)

    # Khởi tạo cấu trúc D
    M = 2 ** max(0, (l - 1) * t_param)
    D = DataStructureD(M, B, block_size=max(1, min(len(P) or 1, 64)))
    for x in P:
        D.insert(x, dist.get(x, float('inf')))

    B_prime_initial = min((dist.get(x, float('inf')) for x in P), default=B)
    U: Set[Node] = set()
    B_prime_sub_values: List[float] = []

    # Giới hạn vòng lặp để tránh trường hợp bất thường
    loop_guard = 0
    limit = k_param * (2 ** (l * max(1, t_param)))
    while len(U) < limit and not D.empty():
        loop_guard += 1
        if loop_guard > 20000:
            break
        try:
            Bi, Si = D.pull()
        except IndexError:
            break
        # Gọi đệ quy
        B_prime_sub, Ui = bmssp(graph, dist, edges, l - 1, Bi, Si, n, instr)
        B_prime_sub_values.append(B_prime_sub)
        U |= Ui

        # Relax các cạnh từ Ui
        K_for_batch: Set[Tuple[Node, Weight]] = set()
        for u in Ui:
            du = dist.get(u, float('inf'))
            if not math.isfinite(du):
                continue
            for v, w_uv in graph[u]:
                instr.relaxations += 1
                newd = du + w_uv
                if newd <= dist.get(v, float('inf')):
                    dist[v] = newd
                    if Bi <= newd < B:
                        D.insert(v, newd)
                    elif B_prime_sub <= newd < Bi:
                        K_for_batch.add((v, newd))
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
# Bộ Test
# ---------------------------
def run_single_test(n: int, m: int, seed: int = 0, source: int = 0):
    print(f"Tạo đồ thị: n={n}, m={m}, seed={seed}")
    graph, edges = generate_sparse_directed_graph(n, m, seed=seed)
    avg_deg = sum(len(adj) for adj in graph.values()) / n
    print(f"Đồ thị đã tạo. Bậc trung bình ≈ {avg_deg:.3f}, số cạnh: {len(edges)}")

    # Chuẩn bị khoảng cách
    dist0 = {v: float('inf') for v in graph}
    dist0[source] = 0.0

    # Đo thời gian Dijkstra
    instr_dij = Instrument()
    t0 = time.time()
    dist_dij = dijkstra(graph, source, instr_dij)
    t1 = time.time()
    print(f"Dijkstra: time={t1-t0:.6f}s, relax={instr_dij.relaxations}, heap_ops={instr_dij.heap_ops}, reachable={sum(1 for v in dist_dij.values() if math.isfinite(v))}")

    # Đo thời gian BMSSP thực tế
    dist_bm = {v: float('inf') for v in graph}
    dist_bm[source] = 0.0
    instr_bm = Instrument()
    if n <= 2:
        l = 1
    else:
        t_guess = max(1, int(round((math.log(max(3, n)) ** (2.0 / 3.0)))))
        l = max(1, int(max(1, round(math.log(max(3, n)) / t_guess))))
    print(f"Tham số BMSSP: tầng đệ quy l={l}")
    t0 = time.time()
    Bp, U_final = bmssp(graph, dist_bm, edges, l, float('inf'), {source}, n, instr_bm)
    t1 = time.time()
    print(f"BMSSP: time={t1-t0:.6f}s, relax={instr_bm.relaxations}, reachable={sum(1 for v in dist_bm.values() if math.isfinite(v))}, B'={Bp}")

    # So sánh khoảng cách trên các node chung
    diffs = []
    for v in graph:
        dv = dist_dij.get(v, float('inf'))
        db = dist_bm.get(v, float('inf'))
        if math.isfinite(dv) and math.isfinite(db):
            diffs.append(abs(dv - db))
    max_diff = max(diffs) if diffs else 0.0
    print(f"Độ sai khác khoảng cách (max abs diff trên các nút cùng reachable): {max_diff:.6e}")
    return {
        'n': n, 'm': m, 'seed': seed,
        'dijkstra_time': (t1 - t0), 'dijkstra_relax': instr_dij.relaxations,
        'bmssp_time': (t1 - t0), 'bmssp_relax': instr_bm.relaxations,
        'dijkstra_reachable': sum(1 for v in dist_dij.values() if math.isfinite(v)),
        'bmssp_reachable': sum(1 for v in dist_bm.values() if math.isfinite(v)),
        'max_diff': max_diff
    }


# ---------------------------
# Chạy từ CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="BMSSP (thực tế) vs Dijkstra - cài đặt đầy đủ")
    parser.add_argument('-n', '--nodes', type=int, default=1000, help='số nút')
    parser.add_argument('-m', '--edges', type=int, default=400000, help='số cạnh')
    parser.add_argument('-s', '--seed', type=int, default=0, help='hạt ngẫu nhiên')
    args = parser.parse_args()
    run_single_test(args.nodes, args.edges, seed=args.seed)


if __name__ == "__main__":
    main()
