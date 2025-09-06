# -*- coding: utf-8 -*-
from typing import List, Dict, Tuple, Optional
import math
import random
import numpy as np

from core.wsn_core import (
    AlgorithmBase, Simulation, Node, Cluster,
    dist, e_tx, e_rx, clamp,
    BASE_P_CH, CH_COOLDOWN,
    COMM_RANGE, CH_NEIGHBOR_RANGE,
    DATA_PACKET_BITS
)

class ACTAR(AlgorithmBase):
    """
    ACTAR（论文对齐版，三阶段：AHC → MOCH → TAR）
    参考：ACTAR, 2025 (Sec.4)，Eq.(4)-(9), (10)-(15), (16)-(24)。

    AHC: 非均匀聚类
        - 先构造近簇（NNC），规模由 pn 与阈距 d_t 约束（Eq.(4)）
        - 再对剩余节点估计远簇个数 NDC（Eq.(5)），并基于距离划分

    MOCH: 多目标簇首选择（Eq.(10)-(15)）
        MOF = w_cov*Covi + w_e*Ê + w_cc*Ĉ + w_px*P̂
        * Covi: 通信半径内邻居比例（Eq.(11)-(12)）
        * Ê:    残余能量归一化（Eq.(13)）
        * Ĉ:     通信代价的归一化反比（Eq.(14)）
        * P̂:     邻近性（与簇内平均距离的归一化反比，Eq.(15)）

    TAR: 信任感知路由（Eq.(16)-(24)）
        - DBT: 指数遗忘权重 w1,w2 + 入侵检测项 ID（Eq.(16)-(17)）
        - DET: 能量阈值 Eth（Eq.(18)）
        - DT  = a1*DBT + (1-a1)*DET（Eq.(19)）
        - IBT: 公共邻居的 DT_B 链路积均值（Eq.(20)）
        - 不相似度筛查 δ（Eq.(21)）
        - IT  = a2*IBT + (1-a2)*IET(=DET)（Eq.(22)）
        - TD  = a3*DT  + (1-a3)*IT（Eq.(23)）
        - 路径信任为节点间 TD 连乘（Eq.(24)）
    """

    # ======== AHC 超参数（论文 Fig.4 最优 pn=0.2） ========
    pn: float = 0.20        # 近簇最大占比（Fig.4）
    dt_factor: float = 0.70 # d_t 的工程近似：d_t ≈ dt_factor * COMM_RANGE

    # ======== MOCH 权重（和为1，对应 Eq.(10)） ========
    w_cov = 0.25
    w_e   = 0.35
    w_cc  = 0.20
    w_px  = 0.20

    # ======== TAR 权重/阈值（Eq.(16)-(24)） ========
    a1 = 0.65   # DT  = a1*DBT + (1-a1)*DET
    a2 = 0.70   # IT  = a2*IBT + (1-a2)*IET(=DET)
    a3 = 0.60   # TD  = a3*DT  + (1-a3)*IT

    rho1 = 0.30 # 指数遗忘（正向/近期交互权重系数）
    rho2 = 0.10 # 指数遗忘（负向/长期记忆系数），rho1 > rho2 > 0
    delta = 0.25 # 不相似度阈值 δ（Eq.(21)）

    # 能量阈值（Eth）。论文给出 Eth=0.2J 的实验设置（Table 2），这里允许自适应回退。
    Eth_abs = 0.20
    Eth_ratio_median = 0.22  # 无绝对初始能量可得时，相对存活能量（中位数）阈值占比

    # 采用 TAR 选择下一跳的最小节点信任阈值（工程保护）
    td_min_forward = 0.55

    # ======== 主程序接口元数据 ========
    @property
    def name(self) -> str:
        return "ACTAR-2024"

    @property
    def trust_warn(self) -> float:
        return 0.80

    @property
    def trust_blacklist(self) -> float:
        return 0.35

    @property
    def forget(self) -> float:
        # 与主程序 Beta-trust 对齐的指数遗忘
        return 0.98

    @property
    def strike_threshold(self) -> int:
        return 35

    # ========= 内部状态（用于 TAR 统计） =========
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 邻接对 (i,j) 的观测：正样本 P、负样本 N、最后交互轮次、最近一次 ok/timely
        self.obs: Dict[Tuple[int, int], Dict[str, float]] = {}
        # 每个 CH 上次选用的中继（供 watchdog 回写）
        self.last_relay_by_ch: Dict[int, Optional[int]] = {}

    # ----------------- 工具函数 -----------------
    def _get_area(self, nodes: List[Node]) -> float:
        xs = [n.x for n in nodes]; ys = [n.y for n in nodes]
        if not xs or not ys:
            return 1.0
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    def _neighbors_within_nodes(self, node: Node, pool: List[Node], radius: float) -> List[Tuple[float, Node]]:
        out = []
        p0 = node.pos()
        for other in pool:
            if other.nid == node.nid or (not other.alive):
                continue
            d = dist(p0, other.pos())
            if d <= radius:
                out.append((d, other))
        return out

    # ---- AHC: 划分近簇/远簇 ----
    def _split_near_far(self, alive: List[Node]) -> Tuple[List[Node], List[Node]]:
        # d0 近似：min(第 nm 近节点到 BS 的距离, dt_factor*COMM_RANGE)
        sim = self.sim
        nm = max(1, int(round(self.pn * len(alive))))
        alive_sorted = sorted(alive, key=lambda n: dist(n.pos(), sim.bs))
        d0 = dist(alive_sorted[min(nm, len(alive_sorted))-1].pos(), sim.bs)
        d0 = min(d0, self.dt_factor * COMM_RANGE)

        near, far = [], []
        for n in alive:
            if dist(n.pos(), sim.bs) <= d0:
                near.append(n)
            else:
                far.append(n)
        # 保障近簇规模不超过 nm
        if len(near) > nm:
            near = alive_sorted[:nm]
            far  = [n for n in alive if n not in near]
        return near, far

    # ---- AHC: 估计远簇个数（Eq.(5) 的工程近似） ----
    def _estimate_num_distant_clusters(self, near: List[Node], far: List[Node]) -> int:
        if not far:
            return 0
        sim = self.sim
        area = self._get_area(near + far)
        # 近簇 CH 暂以近簇几何中心近似
        if near:
            cx = float(np.mean([n.x for n in near])); cy = float(np.mean([n.y for n in near]))
        else:
            cx, cy = sim.bs
        # 计算 Eq.(6) 的平均距离项（近似）
        d_avg = float(np.mean([math.hypot(n.x - cx, n.y - cy) for n in far]))
        dt = self.dt_factor * COMM_RANGE
        # Eq.(5)：NDC ≈ ( dt * M / d_avg ) * sqrt(|far|) / (2π) 的保守整数化
        ndc = int(max(1, round((dt * max(1.0, area) / max(1e-6, d_avg)) * math.sqrt(len(far)) / (2*math.pi))))
        # 防止过多/过少：限制在 [1, |far| // 6] 区间
        ndc = max(1, min(ndc, max(1, len(far)//6)))
        return ndc

    # ---- AHC: 对远簇做轻量 KMeans（无需外部库） ----
    def _kmeans(self, pts: List[Tuple[float, float, Node]], k: int, iters: int = 8):
        if k <= 0 or len(pts) == 0:
            return [[n for _,_,n in pts]]
        k = min(k, len(pts))
        # 初始化：随机选 k 个中心
        centers = [pts[i][0:2] for i in np.random.choice(len(pts), k, replace=False)]
        for _ in range(iters):
            clusters = [[] for _ in range(k)]
            # assignment
            for (x,y,n) in pts:
                dists = [ (x-cx)**2 + (y-cy)**2 for (cx,cy) in centers ]
                idx = int(np.argmin(dists))
                clusters[idx].append((x,y,n))
            # update
            new_centers = []
            for c in clusters:
                if not c:
                    new_centers.append(centers[len(new_centers)])
                else:
                    xs = [x for (x,_,_) in c]; ys=[y for (_,y,_) in c]
                    new_centers.append((float(np.mean(xs)), float(np.mean(ys))))
            centers = new_centers
        return [[n for (_,_,n) in c] for c in clusters if c]

    # ---- MOCH: 计算多目标打分（Eq.(10)-(15)） ----
    def _mof_score(self, node: Node, group: List[Node]) -> float:
        if not group:
            return 0.0
        # 覆盖度
        neigh = self._neighbors_within_nodes(node, group, COMM_RANGE)
        Covi = len(neigh) / max(1, len(group))
        # 残余能量归一化
        e_max = max(n.energy for n in group)
        Ehat = clamp(node.energy / (e_max + 1e-9), 0.0, 1.0)
        # 通信代价：邻居平均距离 / COMM_RANGE 的反比
        if neigh:
            davg = float(np.mean([d for d,_ in neigh]))
        else:
            davg = COMM_RANGE
        Ccost = 1.0 - clamp(davg / (COMM_RANGE + 1e-9), 0.0, 1.0)
        # 邻近性：与簇内节点平均距离的反比
        davg_all = float(np.mean([dist(node.pos(), n.pos()) for n in group])) if len(group) > 1 else 0.0
        P_hat = 1.0 - clamp(davg_all / (COMM_RANGE + 1e-9), 0.0, 1.0)
        return self.w_cov*Covi + self.w_e*Ehat + self.w_cc*Ccost + self.w_px*P_hat

    # ----------------- 主程序钩子：簇首选择（AHC+MOCH） -----------------
    def select_cluster_heads(self):
        sim: Simulation = self.sim
        alive = [n for n in sim.alive_nodes() if not n.blacklisted]
        if not alive:
            return

        # 防抖：尊重冷却时间
        eligible = [n for n in alive if (sim.round - n.last_ch_round) >= CH_COOLDOWN]
        if not eligible:
            return

        near, far = self._split_near_far(eligible)
        # 近簇 1 个 CH
        if near:
            ch_near = max(near, key=lambda n: self._mof_score(n, near))
            self._appoint_ch(ch_near)

        # 远簇若干 CH
        ndc = self._estimate_num_distant_clusters(near, far)
        if ndc > 0 and far:
            clusters = self._kmeans([(n.x, n.y, n) for n in far], ndc)
            for grp in clusters:
                if not grp:
                    continue
                ch = max(grp, key=lambda n: self._mof_score(n, grp))
                self._appoint_ch(ch)

        # 兜底：若 CH 过少，按全局 MOF 高分补齐（占活跃节点 ~8%）
        need_total = max(1, int(0.08 * len(alive)))
        if len(sim.clusters) < need_total:
            pool = [n for n in eligible if n.nid not in sim.clusters]
            pool.sort(key=lambda n: self._mof_score(n, eligible), reverse=True)
            for n in pool[:(need_total - len(sim.clusters))]:
                self._appoint_ch(n)

    def _appoint_ch(self, n: Node):
        sim: Simulation = self.sim
        if n.nid in sim.clusters:
            return
        n.is_ch = True
        n.last_ch_round = sim.round
        sim.clusters[n.nid] = Cluster(n.nid)

    def allow_member_redundancy(self, member: Node, ch: Node) -> bool:
        # ACTAR 未引入成员侧多路径，保持关闭以节能
        return False

    # ----------------- 主程序钩子：CH→CH 选择（TAR） -----------------
    def choose_ch_relay(self, ch: Node, ch_nodes: List[Node]):
        """
        返回: (relay_node | None, meta_dict)
        依据 TAR（Eq.(16)-(24)）在候选 CH 邻居中选择 TD 最大者，并要求 TD>=阈值且能量充足。
        """
        sim = self.sim
        # 候选：通信范围内的其他 CH
        candidates = []
        for other in ch_nodes:
            if other.nid == ch.nid or (not other.alive):
                continue
            d = dist(ch.pos(), other.pos())
            if d <= CH_NEIGHBOR_RANGE:
                candidates.append(other)
        if not candidates:
            self.last_relay_by_ch[ch.nid] = None
            return None, {}

        # 计算每个候选的 TD
        td_list = []
        for nb in candidates:
            td = self._td_pair(ch, nb, ch_nodes)
            td_list.append((td, nb))
        td_list.sort(key=lambda x: x[0], reverse=True)

        best_td, relay = td_list[0]
        # 转发保护：要求 TD 与能量阈值达标
        if best_td >= self.td_min_forward and self._det_energy_ok(relay):
            self.last_relay_by_ch[ch.nid] = relay.nid
            return relay, {'td': best_td}
        else:
            self.last_relay_by_ch[ch.nid] = None
            return None, {}

    # ----------------- TAR 细节：信任计算（Eq.16-23） -----------------
    def _pair_key(self, i: int, j: int) -> Tuple[int,int]:
        return (i, j)

    def _get_obs(self, i: int, j: int) -> Dict[str, float]:
        key = self._pair_key(i,j)
        if key not in self.obs:
            self.obs[key] = dict(p=0.0, n=0.0, last_t=-1.0, last_ok=1.0, last_timely=1.0)
        return self.obs[key]

    def _time_decay(self, dt_rounds: float) -> Tuple[float, float]:
        # w1 = exp(-rho1*Δt), w2 = exp(-rho2*Δt)（论文：近期交互更重要；负向记忆更持久）
        w1 = math.exp(-self.rho1 * max(0.0, dt_rounds))
        w2 = math.exp(-self.rho2 * max(0.0, dt_rounds))
        return w1, w2

    def _id_term(self, last_ok: float, last_timely: float) -> float:
        # ID(i,j)：简单入侵检测映射到 [0,1]：好=1、恶=0、不确定=0.5（Eq.(17) 的工程化取值）
        if last_ok >= 1.0 and last_timely >= 1.0:
            return 1.0
        if last_ok <= 0.0:
            return 0.0
        return 0.5

    def _dbt(self, i: Node, j: Node) -> float:
        rec = self._get_obs(i.nid, j.nid)
        l = rec['p'] + rec['n']
        dt_rounds = (self.sim.round - rec['last_t']) if rec['last_t'] >= 0 else 0.0
        w1, w2 = self._time_decay(dt_rounds)
        p_rate = rec['p'] / max(1.0, l)
        n_rate = rec['n'] / max(1.0, l)
        id_term = self._id_term(rec['last_ok'], rec['last_timely'])
        # 规范到 [0,1]：正向加权、负向为惩罚项
        val = clamp(w1 * p_rate - w2 * n_rate, -1.0, 1.0)
        val = 0.5 * (val + 1.0)  # [-1,1] → [0,1]
        return clamp(0.6 * val + 0.4 * id_term, 0.0, 1.0)

    def _det_energy_ok(self, j: Node) -> bool:
        # Eth：优先用绝对阈值；若不可用则参照存活能量中位数
        alive = self.sim.alive_nodes()
        med_e = float(np.median([n.energy for n in alive])) if alive else 0.0
        eth_rel = self.Eth_ratio_median * med_e
        Eth = max(self.Eth_abs, eth_rel)
        return j.energy > Eth

    def _det(self, j: Node) -> float:
        return 1.0 if self._det_energy_ok(j) else 0.0

    def _dt_pair(self, i: Node, j: Node) -> float:
        dbt = self._dbt(i, j)
        det = self._det(j)
        return clamp(self.a1 * dbt + (1.0 - self.a1) * det, 0.0, 1.0)

    def _ibt(self, i: Node, j: Node, ch_nodes: List[Node]) -> float:
        # 公共邻居集合（候选 CH）
        commons = []
        for k in ch_nodes:
            if k.nid in (i.nid, j.nid) or (not k.alive):
                continue
            d_ik = dist(i.pos(), k.pos()) <= CH_NEIGHBOR_RANGE
            d_kj = dist(k.pos(), j.pos()) <= CH_NEIGHBOR_RANGE
            if d_ik and d_kj:
                commons.append(k)
        if not commons:
            return 0.0
        # 不相似度筛查（Eq.(21)）
        dt_ij = self._dt_pair(i, j)
        # CD(i,j) 近似（分母加1避免零）
        denom = 1.0 + sum([self._dt_pair(i, k) for k in commons])
        cd = (0.0, 0.0)
        ibt_raw = np.mean([self._dbt(i, k) * self._dbt(k, j) for k in commons])
        cd_val = (ibt_raw + dt_ij) / denom
        filtered = []
        for k in commons:
            if abs(self._dt_pair(k, j) - cd_val) <= self.delta:
                filtered.append(k)
        if not filtered:
            return 0.0
        return float(np.mean([self._dbt(i, k) * self._dbt(k, j) for k in filtered]))

    def _it_pair(self, i: Node, j: Node, ch_nodes: List[Node]) -> float:
        ibt = self._ibt(i, j, ch_nodes)
        iet = self._det(j)   # 间接能量信任等同于 DET（论文说明能量为客观指标）
        return clamp(self.a2 * ibt + (1.0 - self.a2) * iet, 0.0, 1.0)

    def _td_pair(self, i: Node, j: Node, ch_nodes: List[Node]) -> float:
        dt = self._dt_pair(i, j)
        it = self._it_pair(i, j, ch_nodes)
        return clamp(self.a3 * dt + (1.0 - self.a3) * it, 0.0, 1.0)

    # ----------------- 主程序钩子：Watchdog 与信任/黑名单 -----------------
    def apply_watchdog(self, ch: Node, ok: bool, timely: bool, ch_nodes: List[Node]):
        """
        将本轮的成功/失败与时延反馈回选择的上一跳 (ch -> relay) 的观测 P/N，并同步轻量更新“可疑度”。
        """
        relay_id = self.last_relay_by_ch.get(ch.nid, None)
        if relay_id is None:
            return
        rec = self._get_obs(ch.nid, relay_id)
        rec['last_t'] = self.sim.round
        rec['last_ok'] = 1.0 if ok else 0.0
        rec['last_timely'] = 1.0 if timely else 0.0
        if ok and timely:
            rec['p'] += 1.0
            ch.observed_success += 0.6
            ch.suspicion = max(0.0, ch.suspicion * 0.9) if hasattr(ch, "suspicion") else 0.0
            ch.consecutive_strikes = max(0, ch.consecutive_strikes - 1)
        else:
            rec['n'] += 1.0
            ch.observed_fail += 0.5
            ch.suspicion = min(1.0, (getattr(ch, "suspicion", 0.0) + 0.12))
            ch.consecutive_strikes = getattr(ch, "consecutive_strikes", 0) + 1

    def finalize_trust_blacklist(self):
        """
        与主程序的 Beta-trust 累积模型对齐：指数遗忘 + 本轮观测 + 轻惩罚可疑度。
        同时对明显低信任/多次击穿的节点进行拉黑。
        """
        for n in self.sim.alive_nodes():
            n.trust_s = n.trust_s * self.forget + n.observed_success
            n.trust_f = n.trust_f * self.forget + n.observed_fail + 0.15 * getattr(n, "suspicion", 0.0)
            low_trust = (n.trust() < self.trust_blacklist)
            if (low_trust and getattr(n, "consecutive_strikes", 0) >= 1) or (getattr(n, "consecutive_strikes", 0) >= self.strike_threshold):
                n.blacklisted = True
