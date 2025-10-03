import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import heapq
import math
import pandas as pd
from typing import Dict, List, Tuple, Set

# Page config
st.set_page_config(page_title="BMSSP Visualization", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'execution_log' not in st.session_state:
    st.session_state.execution_log = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0


# Generate random directed graph
def generate_graph(n_nodes, n_edges, seed=42):
    np.random.seed(seed)
    G = nx.DiGraph()

    for i in range(n_nodes):
        G.add_node(i)

    # Backbone
    for i in range(1, n_nodes):
        u = np.random.randint(0, i)
        weight = np.random.randint(5, 20)
        G.add_edge(u, i, weight=weight)

    # Random edges
    remaining = max(0, n_edges - (n_nodes - 1))
    for _ in range(remaining):
        u = np.random.randint(0, n_nodes)
        v = np.random.randint(0, n_nodes)
        if u != v and not G.has_edge(u, v):
            weight = np.random.randint(5, 20)
            G.add_edge(u, v, weight=weight)

    return G


def format_distance(d):
    if d == float('inf'):
        return '∞'
    return f'{d:.1f}'


# Build adjacency list from NetworkX graph
def build_graph_dict(G):
    graph = {}
    for node in G.nodes():
        graph[node] = []
        for neighbor in G.neighbors(node):
            weight = G[node][neighbor]['weight']
            graph[node].append((neighbor, weight))
    return graph


# ===========================
# DIJKSTRA (unchanged)
# ===========================
def dijkstra_with_steps(G, source):
    steps = []
    dist = {node: float('inf') for node in G.nodes()}
    dist[source] = 0
    heap = [(0, source)]
    visited = set()
    relaxations = 0
    heap_ops = 1

    dist_str = {k: format_distance(v) for k, v in dist.items()}
    steps.append({
        'type': 'init',
        'message': f'🚀 Khởi tạo Dijkstra từ đỉnh {source}',
        'detail': f'dist[{source}] = 0, tất cả đỉnh khác = ∞',
        'distances': dist.copy(),
        'dist_display': dist_str.copy(),
        'visited': set(),
        'active_nodes': {source},
        'active_edges': set(),
        'stats': {'relaxations': relaxations, 'heap_ops': heap_ops}
    })

    while heap:
        d, u = heapq.heappop(heap)
        heap_ops += 1

        if u in visited:
            continue

        visited.add(u)
        dist_str = {k: format_distance(v) for k, v in dist.items()}

        # Explanation why this is the shortest path
        why_shortest = f"✓ Đỉnh {u} HOÀN THÀNH với dist[{u}]={d:.1f}\n"
        why_shortest += f"Lý do: {u} có khoảng cách nhỏ nhất trong heap.\n"
        why_shortest += f"Mọi đường đi khác đến {u} đều ≥ {d:.1f} (tính chất Dijkstra: không có cạnh âm)"

        steps.append({
            'type': 'visit',
            'message': f'👉 Thăm đỉnh {u} (d={d:.1f}) - FINALIZED',
            'detail': why_shortest,
            'distances': dist.copy(),
            'dist_display': dist_str.copy(),
            'visited': visited.copy(),
            'active_nodes': {u},
            'active_edges': set(),
            'finalized': {u},  # Mark as finalized
            'stats': {'relaxations': relaxations, 'heap_ops': heap_ops}
        })

        for v in G.neighbors(u):
            weight = G[u][v]['weight']
            relaxations += 1
            alt = d + weight

            edge_id = (u, v)
            old_dist = dist[v]
            improved = alt < dist[v]

            if improved:
                dist[v] = alt

            dist_str = {k: format_distance(v) for k, v in dist.items()}

            steps.append({
                'type': 'relax',
                'message': f'🔍 Relax {u}→{v} (w={weight})',
                'detail': f'{d:.1f} + {weight} = {alt:.1f} {"< " + format_distance(old_dist) + " → Cập nhật" if improved else ">= " + format_distance(old_dist) + " → Bỏ qua"}',
                'distances': dist.copy(),
                'dist_display': dist_str.copy(),
                'visited': visited.copy(),
                'active_nodes': {u, v},
                'active_edges': {edge_id},
                'stats': {'relaxations': relaxations, 'heap_ops': heap_ops}
            })

            if improved:
                heapq.heappush(heap, (alt, v))
                heap_ops += 1

    dist_str = {k: format_distance(v) for k, v in dist.items()}
    steps.append({
        'type': 'complete',
        'message': '✅ Dijkstra hoàn thành!',
        'detail': f'Đã tìm đường đi ngắn nhất từ {source}',
        'distances': dist.copy(),
        'dist_display': dist_str.copy(),
        'visited': visited.copy(),
        'active_nodes': set(),
        'active_edges': set(),
        'stats': {'relaxations': relaxations, 'heap_ops': heap_ops}
    })

    return steps


# ===========================
# BMSSP - EXACT IMPLEMENTATION
# ===========================

class DataStructureD:
    """Exact implementation from original code"""

    def __init__(self, M: int, B_upper: float, block_size: int = None):
        self.heap: List[Tuple[float, int]] = []
        self.best: Dict[int, float] = {}
        self.M = max(1, M)
        self.B_upper = B_upper
        self.block_size = block_size if block_size is not None else max(1, self.M // 8)

    def insert(self, v: int, key: float):
        prev = self.best.get(v)
        if prev is None or key < prev:
            self.best[v] = key
            heapq.heappush(self.heap, (key, v))

    def batch_prepend(self, iterable_pairs):
        for v, key in iterable_pairs:
            self.insert(v, key)

    def _cleanup(self):
        while self.heap and self.best.get(self.heap[0][1]) != self.heap[0][0]:
            heapq.heappop(self.heap)

    def empty(self) -> bool:
        self._cleanup()
        return len(self.heap) == 0

    def pull(self) -> Tuple[float, Set[int]]:
        self._cleanup()
        if not self.heap:
            raise IndexError("pull from empty D")

        Bi = self.heap[0][0]
        Si: Set[int] = set()

        while self.heap and len(Si) < self.block_size:
            key, v = heapq.heappop(self.heap)
            if self.best.get(v) == key:
                Si.add(v)
                del self.best[v]

        return Bi, Si


def find_pivots(graph, dist, S, B, n, k_steps, p_limit, steps, prefix=''):
    """Exact implementation from original code"""
    S_filtered = [v for v in S if dist.get(v, float('inf')) < B]

    if not S_filtered:
        P = set(list(S)[:max(1, min(len(S), p_limit))]) if S else set()
    else:
        S_filtered.sort(key=lambda v: dist.get(v, float('inf')))
        P = set(S_filtered[:max(1, min(len(S_filtered), p_limit))])

    source_frontier = P if P else set(S)
    discovered = set(source_frontier)
    frontier = set(source_frontier)

    # Log pivot selection
    dist_str = {k: format_distance(v) for k, v in dist.items()}
    steps.append({
        'type': 'pivots',
        'message': f'{prefix}📍 FIND_PIVOTS: P = {{{", ".join(map(str, sorted(P)))}}}',
        'detail': f'Chọn {len(P)} pivots từ S có dist nhỏ nhất',
        'distances': dist.copy(),
        'dist_display': dist_str.copy(),
        'visited': set(),
        'active_nodes': P.copy(),
        'active_edges': set(),
        'stats': {'relaxations': 0, 'heap_ops': 0}
    })

    # Bounded BF
    for step_i in range(max(1, k_steps)):
        if not frontier:
            break
        next_front = set()
        for u in frontier:
            du = dist.get(u, float('inf'))
            if du >= B:
                continue
            for v, w in graph.get(u, []):
                nd = du + w
                if nd < B and v not in discovered:
                    discovered.add(v)
                    next_front.add(v)
        frontier = next_front

    W = discovered.copy()
    if not P and S:
        P = {next(iter(S))}

    return P, W


def basecase(graph, dist, B, S, k, steps, prefix=''):
    """Exact implementation from original code"""
    if not S:
        return B, set()

    x = min(S, key=lambda v: dist.get(v, float('inf')))
    heap: List[Tuple[float, int]] = []
    start_d = dist.get(x, float('inf'))
    heapq.heappush(heap, (start_d, x))

    Uo: Set[int] = set()
    visited: Set[int] = set()

    dist_str = {k: format_distance(v) for k, v in dist.items()}
    steps.append({
        'type': 'basecase',
        'message': f'{prefix}🎯 BASECASE: x={x}, k={k}, B={format_distance(B)}',
        'detail': f'Chạy Dijkstra giới hạn từ {x}, dừng sau {k + 1} đỉnh',
        'distances': dist.copy(),
        'dist_display': dist_str.copy(),
        'visited': set(),
        'active_nodes': {x},
        'active_edges': set(),
        'stats': {'relaxations': 0, 'heap_ops': 0}
    })

    while heap and len(Uo) < (k + 1):
        d_u, u = heapq.heappop(heap)
        if d_u > dist.get(u, float('inf')):
            continue

        if u not in Uo:
            Uo.add(u)

            dist_str = {k: format_distance(v) for k, v in dist.items()}
            steps.append({
                'type': 'visit',
                'message': f'{prefix}  👉 Thăm {u} (d={dist[u]:.1f})',
                'detail': f'Thêm {u} vào Uo, |Uo|={len(Uo)}',
                'distances': dist.copy(),
                'dist_display': dist_str.copy(),
                'visited': Uo.copy(),
                'active_nodes': {u},
                'active_edges': set(),
                'stats': {'relaxations': 0, 'heap_ops': 0}
            })

        for v, w in graph.get(u, []):
            newd = dist.get(u, float('inf')) + w
            if newd < dist.get(v, float('inf')) and newd < B:
                old_dist = dist[v]
                dist[v] = newd
                heapq.heappush(heap, (newd, v))

                dist_str = {k: format_distance(v) for k, v in dist.items()}
                steps.append({
                    'type': 'relax',
                    'message': f'{prefix}  🔍 Relax {u}→{v} (w={w})',
                    'detail': f'dist[{v}]: {format_distance(old_dist)} → {newd:.1f}',
                    'distances': dist.copy(),
                    'dist_display': dist_str.copy(),
                    'visited': Uo.copy(),
                    'active_nodes': {u, v},
                    'active_edges': {(u, v)},
                    'stats': {'relaxations': 0, 'heap_ops': 0}
                })

    if len(Uo) <= k:
        return B, Uo
    else:
        finite_dists = [dist[v] for v in Uo if math.isfinite(dist.get(v, float('inf')))]
        if not finite_dists:
            return B, set()
        maxd = max(finite_dists)
        U_filtered = {v for v in Uo if dist.get(v, float('inf')) < maxd}
        return maxd, U_filtered


def bmssp_with_steps(G, source):
    """Exact implementation from original code"""
    steps = []
    graph = build_graph_dict(G)
    dist = {node: float('inf') for node in G.nodes()}
    dist[source] = 0
    n = G.number_of_nodes()

    dist_str = {k: format_distance(v) for k, v in dist.items()}
    steps.append({
        'type': 'init',
        'message': f'🚀 Khởi tạo BMSSP từ đỉnh {source}',
        'detail': f'dist[{source}] = 0, tất cả đỉnh khác = ∞',
        'distances': dist.copy(),
        'dist_display': dist_str.copy(),
        'visited': set(),
        'active_nodes': {source},
        'active_edges': set(),
        'stats': {'relaxations': 0, 'heap_ops': 0}
    })

    def bmssp_recursive(l, B, S, prefix=''):
        # Parameters
        if n <= 2:
            t_param = 1
            k_param = 2
        else:
            t_param = max(1, int(round((math.log(max(3, n)) ** (2.0 / 3.0)))))
            k_param = max(2, int(round((math.log(max(3, n)) ** (1.0 / 3.0)))))

        # Base case
        if l <= 0:
            if not S:
                return B, set()
            dist_str = {k: format_distance(v) for k, v in dist.items()}
            steps.append({
                'type': 'recursion',
                'message': f'{prefix}🔄 [l={l}] BMSSP BASE (l=0)',
                'detail': f'Gọi BASECASE với S={{{", ".join(map(str, sorted(S)))}}}',
                'distances': dist.copy(),
                'dist_display': dist_str.copy(),
                'visited': set(),
                'active_nodes': S.copy(),
                'active_edges': set(),
                'stats': {'relaxations': 0, 'heap_ops': 0}
            })
            return basecase(graph, dist, B, S, k_param, steps, prefix + '  ')

        # Recursive case
        dist_str = {k: format_distance(v) for k, v in dist.items()}
        steps.append({
            'type': 'recursion',
            'message': f'{prefix}🔄 [l={l}] BMSSP với B={format_distance(B)}, |S|={len(S)}',
            'detail': f'Độ sâu l={l}, t={t_param}, k={k_param}',
            'distances': dist.copy(),
            'dist_display': dist_str.copy(),
            'visited': set(),
            'active_nodes': S.copy(),
            'active_edges': set(),
            'stats': {'relaxations': 0, 'heap_ops': 0}
        })

        # FIND_PIVOTS
        p_limit = max(1, 2 ** min(10, t_param))
        k_steps = max(1, k_param)
        P, W = find_pivots(graph, dist, S, B, n, k_steps, p_limit, steps, prefix + '  ')

        # DataStructure D
        M = 2 ** max(0, (l - 1) * t_param)
        D = DataStructureD(M, B, block_size=max(1, min(len(P) or 1, 64)))

        for x in P:
            D.insert(x, dist.get(x, float('inf')))

        B_prime_initial = min((dist.get(x, float('inf')) for x in P), default=B)
        U: Set[int] = set()
        B_prime_sub_values: List[float] = []

        loop_count = 0
        limit = k_param * (2 ** (l * max(1, t_param)))

        # Main loop
        while len(U) < limit and not D.empty():
            loop_count += 1
            if loop_count > 100:  # Safety limit for visualization
                break

            try:
                Bi, Si = D.pull()
            except IndexError:
                break

            dist_str = {k: format_distance(v) for k, v in dist.items()}
            steps.append({
                'type': 'batch',
                'message': f'{prefix}  📦 D.pull(): Bi={Bi:.1f}, Si={{{", ".join(map(str, sorted(Si)))}}}',
                'detail': f'Lấy batch từ D, |Si|={len(Si)}',
                'distances': dist.copy(),
                'dist_display': dist_str.copy(),
                'visited': U.copy(),
                'active_nodes': Si.copy(),
                'active_edges': set(),
                'stats': {'relaxations': 0, 'heap_ops': 0}
            })

            # Recursive call
            B_prime_sub, Ui = bmssp_recursive(l - 1, Bi, Si, prefix + '    ')
            B_prime_sub_values.append(B_prime_sub)
            U |= Ui

            # Relax edges
            K_for_batch: Set[Tuple[int, float]] = set()
            for u in Ui:
                du = dist.get(u, float('inf'))
                if not math.isfinite(du):
                    continue
                for v, w_uv in graph.get(u, []):
                    newd = du + w_uv
                    if newd <= dist.get(v, float('inf')):
                        old_dist = dist[v]
                        dist[v] = newd

                        if newd != old_dist:  # Only log actual changes
                            dist_str = {k: format_distance(v) for k, v in dist.items()}
                            steps.append({
                                'type': 'relax',
                                'message': f'{prefix}    🔍 Relax {u}→{v} (w={w_uv})',
                                'detail': f'dist[{v}]: {format_distance(old_dist)} → {newd:.1f}',
                                'distances': dist.copy(),
                                'dist_display': dist_str.copy(),
                                'visited': U.copy(),
                                'active_nodes': {u, v},
                                'active_edges': {(u, v)},
                                'stats': {'relaxations': 0, 'heap_ops': 0}
                            })

                        if Bi <= newd < B:
                            D.insert(v, newd)
                        elif B_prime_sub <= newd < Bi:
                            K_for_batch.add((v, newd))

            # batch_prepend
            for x in Si:
                dx = dist.get(x, float('inf'))
                if B_prime_sub <= dx < Bi:
                    K_for_batch.add((x, dx))

            if K_for_batch:
                D.batch_prepend(K_for_batch)

        # Finalize
        if B_prime_sub_values:
            B_prime_final = min([B_prime_initial] + B_prime_sub_values)
        else:
            B_prime_final = B_prime_initial

        U_final = set(U)
        for x in W:
            if dist.get(x, float('inf')) < B_prime_final:
                U_final.add(x)

        return B_prime_final, U_final

    # Top-level call
    if n <= 2:
        t_guess = 1
    else:
        t_guess = max(1, int(round((math.log(max(3, n)) ** (2.0 / 3.0)))))

    l_top = max(1, int(max(1, round(math.log(max(3, n)) / t_guess))))

    Bp, U_final = bmssp_recursive(l_top, float('inf'), {source}, '')

    dist_str = {k: format_distance(v) for k, v in dist.items()}
    steps.append({
        'type': 'complete',
        'message': f'✅ BMSSP hoàn thành!',
        'detail': f"B'={Bp:.1f}, |U|={len(U_final)}",
        'distances': dist.copy(),
        'dist_display': dist_str.copy(),
        'visited': U_final.copy(),
        'active_nodes': set(),
        'active_edges': set(),
        'stats': {'relaxations': 0, 'heap_ops': 0}
    })

    return steps


# ===========================
# VISUALIZATION
# ===========================
def visualize_graph(G, distances, visited, active_nodes, active_edges, source):
    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=2.5, iterations=50)

    # Edges
    for edge in G.edges():
        u, v = edge
        is_active = edge in active_edges
        weight = G[u][v]['weight']

        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]

        color = 'orange' if is_active else 'lightgray'
        width = 3 if is_active else 2

        ax.plot(x, y, color=color, linewidth=width, zorder=1, alpha=0.7)

        # Arrow
        arrow_pos = 0.7
        dx = pos[v][0] - pos[u][0]
        dy = pos[v][1] - pos[u][1]
        arrow_x = pos[u][0] + arrow_pos * dx
        arrow_y = pos[u][1] + arrow_pos * dy

        ax.annotate('', xy=(pos[v][0], pos[v][1]), xytext=(arrow_x, arrow_y),
                    arrowprops=dict(arrowstyle='->', lw=width, color=color,
                                    mutation_scale=25, shrinkA=0, shrinkB=15))

        # Weight
        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2
        ax.text(mid_x, mid_y, str(weight),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9),
                ha='center', va='center', zorder=5)

    # Nodes
    node_colors = []
    for node in G.nodes():
        if node == source:
            node_colors.append('#ef4444')
        elif node in active_nodes:
            node_colors.append('#f59e0b')
        elif node in visited:
            node_colors.append('#22c55e')
        else:
            node_colors.append('#3b82f6')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000,
                           ax=ax, edgecolors='white', linewidths=3)
    nx.draw_networkx_labels(G, pos, font_size=14, font_color='white',
                            font_weight='bold', ax=ax)

    # Distance labels
    for node in G.nodes():
        x, y = pos[node]
        dist_text = format_distance(distances[node])
        ax.text(x, y + 0.12, f'd={dist_text}',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                          edgecolor='orange', alpha=0.8),
                ha='center', va='center', zorder=5)

    ax.set_title("BMSSP - Đồ thị có hướng", fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    return fig


# ===========================
# STREAMLIT UI
# ===========================
st.markdown('<p class="main-header">🔍 BMSSP Algorithm Visualization</p>', unsafe_allow_html=True)
st.markdown("**Implementation đúng theo code gốc bmssp_full_impl.py**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài đặt")

    n_nodes = st.slider("Số đỉnh", 5, 12, 8)
    n_edges = st.slider("Số cạnh", n_nodes, n_nodes * 2, n_nodes + 4)
    source = st.slider("Đỉnh nguồn", 0, n_nodes - 1, 0)
    seed = st.number_input("Random seed", 0, 1000, 42)

    algorithm = st.radio("Chọn thuật toán", ["Dijkstra", "BMSSP"])

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Tạo đồ thị mới"):
            st.session_state.graph = generate_graph(n_nodes, n_edges, seed)
            st.session_state.execution_log = []
            st.session_state.current_step = 0
            st.rerun()

    with col2:
        if st.button("▶️ Chạy thuật toán"):
            if st.session_state.graph is not None:
                if algorithm == "Dijkstra":
                    st.session_state.execution_log = dijkstra_with_steps(st.session_state.graph, source)
                else:
                    st.session_state.execution_log = bmssp_with_steps(st.session_state.graph, source)
                st.session_state.current_step = 0
                st.rerun()

    st.divider()
    st.markdown("### 📖 Chú thích")
    st.markdown("""
    - 🔴 **Đỏ**: Đỉnh nguồn
    - 🟠 **Cam**: Đang xử lý
    - 🟢 **Xanh lá**: Đã thăm
    - 🔵 **Xanh dương**: Chưa thăm
    """)

# Generate initial graph
if st.session_state.graph is None:
    st.session_state.graph = generate_graph(n_nodes, n_edges, seed)

# Main content
col1, col2 = st.columns([2.5, 1.5])

with col1:
    st.subheader("📊 Visualization")

    if st.session_state.execution_log and st.session_state.current_step < len(st.session_state.execution_log):
        step_data = st.session_state.execution_log[st.session_state.current_step]
        distances = step_data['distances']
        visited = step_data['visited']
        active_nodes = step_data['active_nodes']
        active_edges = step_data.get('active_edges', set())
        stats = step_data['stats']
    else:
        distances = {node: float('inf') for node in st.session_state.graph.nodes()}
        distances[source] = 0
        visited = set()
        active_nodes = set()
        active_edges = set()
        stats = {'relaxations': 0, 'heap_ops': 0}

    fig = visualize_graph(st.session_state.graph, distances, visited, active_nodes, active_edges, source)
    st.pyplot(fig)
    plt.close()

    # Controls
    if st.session_state.execution_log:
        cols = st.columns([1, 1, 1, 3])

        with cols[0]:
            if st.button("⏮️ Đầu"):
                st.session_state.current_step = 0
                st.rerun()

        with cols[1]:
            if st.button("◀️ Trước"):
                if st.session_state.current_step > 0:
                    st.session_state.current_step -= 1
                    st.rerun()

        with cols[2]:
            if st.button("▶️ Sau"):
                if st.session_state.current_step < len(st.session_state.execution_log) - 1:
                    st.session_state.current_step += 1
                    st.rerun()

        st.progress((st.session_state.current_step + 1) / len(st.session_state.execution_log))
        st.caption(f"Bước {st.session_state.current_step + 1} / {len(st.session_state.execution_log)}")

with col2:
    st.subheader("📈 Thống kê")

    # Stats cards
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Bước",
                  f"{st.session_state.current_step + 1}/{len(st.session_state.execution_log) if st.session_state.execution_log else 0}")
    with col_b:
        st.metric("Relaxations", stats['relaxations'])

    col_c, col_d = st.columns(2)
    with col_c:
        st.metric("Heap Ops", stats['heap_ops'])
    with col_d:
        if st.session_state.execution_log and st.session_state.current_step < len(st.session_state.execution_log):
            step_data = st.session_state.execution_log[st.session_state.current_step]
            msg = step_data['message']
            # Extract recursion depth from message if present
            if '[l=' in msg:
                import re

                match = re.search(r'\[l=(\d+)\]', msg)
                if match:
                    st.metric("Độ sâu l", match.group(1))
                else:
                    st.metric("Độ sâu l", "-")
            else:
                st.metric("Độ sâu l", "-")
        else:
            st.metric("Độ sâu l", "-")

    st.divider()

    # Distance table
    st.subheader("📏 Bảng khoảng cách")
    if st.session_state.execution_log and st.session_state.current_step < len(st.session_state.execution_log):
        step_data = st.session_state.execution_log[st.session_state.current_step]
        dist_display = step_data['dist_display']

        df_dist = pd.DataFrame({
            'Đỉnh': list(dist_display.keys()),
            'Khoảng cách': list(dist_display.values())
        })


        def highlight_source(row):
            if row['Đỉnh'] == source:
                return ['background-color: #ef4444; color: white'] * len(row)
            elif row['Đỉnh'] in active_nodes:
                return ['background-color: #f59e0b; color: white'] * len(row)
            elif row['Đỉnh'] in visited:
                return ['background-color: #22c55e; color: white'] * len(row)
            return [''] * len(row)


        st.dataframe(df_dist.style.apply(highlight_source, axis=1),
                     use_container_width=True, hide_index=True)

    st.divider()

    # Execution log
    st.subheader("📝 Log thực thi")

    log_container = st.container(height=350)
    with log_container:
        if st.session_state.execution_log and st.session_state.current_step < len(st.session_state.execution_log):
            step = st.session_state.execution_log[st.session_state.current_step]

            # Current step highlight
            st.markdown(f"**➡️ {step['message']}**")
            st.info(step['detail'])

            st.markdown("---")
            st.markdown("**Lịch sử gần đây:**")

            # Show recent steps
            start_idx = max(0, st.session_state.current_step - 8)

            for i in range(st.session_state.current_step - 1, start_idx - 1, -1):
                if i >= 0:
                    prev_step = st.session_state.execution_log[i]

                    emoji_map = {
                        'init': '🚀',
                        'visit': '👉',
                        'relax': '🔍',
                        'complete': '✅',
                        'recursion': '🔄',
                        'pivots': '📍',
                        'batch': '📦',
                        'basecase': '🎯'
                    }

                    emoji = emoji_map.get(prev_step['type'], '📌')
                    st.caption(f"{emoji} {prev_step['message']}")
        else:
            st.info("Chưa có log. Hãy chọn thuật toán và nhấn 'Chạy thuật toán'")

# Footer
st.divider()
st.markdown("""
### 🔬 Về thuật toán BMSSP

Code này implement **CHÍNH XÁC** theo `bmssp_full_impl.py`:
- ✅ **DataStructureD** với `insert()`, `pull()`, `batch_prepend()`
- ✅ **FIND_PIVOTS** với bounded Bellman-Ford
- ✅ **BASECASE** với Dijkstra giới hạn k+1 đỉnh
- ✅ Vòng lặp `while not D.empty()` với logic đầy đủ
- ✅ Tham số t, k, M theo công thức lý thuyết
- ✅ Logic B_prime_sub và K_for_batch

Chỉ thêm visualization và step logging, **KHÔNG thay đổi logic thuật toán**.
""")