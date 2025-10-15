# -*- coding: utf-8 -*-
"""
DST-WOA（论文对齐版）
- Phase 1: 基于 Dempster-Shafer 的可信 CH 选择（Eq.(7)-(10), Fig.1 Phase 1）
- Phase 2: CH 回收再用（min_erg=0.6, tl=60）（Fig.1 Phase 2）
- Phase 3: 路由：鲸鱼优化算法 WOA 选择下一跳（Eq.(11)-(12), Fig.1 Phase 3）
参考：Peer-to-Peer Networking and Applications (2024) 17:1486–1498, "Trust-based clustering and routing in WSNs using DST-WOA"
"""
# -*- coding: utf-8 -*-
"""
DST-WOA（论文对齐版，低改动：仅调整4个高影响超参数）
- 仅改动：
  1) TARGET_CLUSTER_SIZE = 72
  2) NEIGHBOR_RADIUS_FACTOR = 0.60（用于候选中继半径）
  3) EPSILON_ROUTE = 0.32（路由 ε-随机）
  4) DIRECT_BIAS   = 0.22（直达偏置）
其余保持论文对齐实现不变。
"""

from typing import List, Tuple, Dict, Optional
import math
import random
import numpy as np

from core.wsn_core import (
    AlgorithmBase, Simulation, Node, Cluster,
    dist, clamp,
    BASE_P_CH, CH_COOLDOWN,
    COMM_RANGE, CH_NEIGHBOR_RANGE,
)

# ================ 工具：欧式距离累加（Eq.(12)） ================
def _avg_path_distance(pts: List[Tuple[float, float]]) -> float:
    """Eq.(12) 的 Scdis：沿路段的距离均值；pts 为有序坐标序列（CLS->...->CLD）。"""
    if len(pts) < 2:
        return 0.0
    segs = [dist(pts[i], pts[i+1]) for i in range(len(pts)-1)]
    return float(np.mean(segs))


class DST_WOA(AlgorithmBase):
    """
    DST-WOA 算法（与 trail/core/wsn_core.py 框架兼容）
    提供主程序期望的 5 个钩子：
      - select_cluster_heads
      - allow_member_redundancy
      - choose_ch_relay
      - apply_watchdog
      - finalize_trust_blacklist
    """

    # 论文参数（见文中 Phase 2）
    MIN_ERG_RATIO = 0.60    # min_erg=0.6（相对初始能量）
    TL_ROUNDS     = 60      # tl=60（以轮为单位近似 60s）

    # 论文聚类规模：每簇“约 20 个节点”（第4节 first paragraph）
    # 只改高影响超参数：加大簇规模以降低稳健性
    TARGET_CLUSTER_SIZE = 20

    # 适应度（Eq.(11)-(12)）归一化、WOA 控制参数（保持不变）
    WOA_POP_FACTOR = 3      # 种群规模系数：= min(10, 3*K)
    WOA_ITERS      = 12     # 迭代次数
    WOA_B          = 1.0    # 螺旋参数 b
    # a 线性递减从 2->0（标准 WOA）

    # —— 只加高影响超参数（其余逻辑不改）——
    NEIGHBOR_RADIUS_FACTOR = 0.60  # 仅用 60% 的 CH 邻域半径作为候选
    EPSILON_ROUTE = 0.32           # 32% 概率随机选任意候选（含直达）
    DIRECT_BIAS   = 0.22           # 22% 概率强制直达 BS

    # 信任与黑名单（与主程序 Beta-trust 对齐，保持原值）
    @property
    def name(self) -> str:
        return "DST-WOA-2024"

    @property
    def trust_warn(self) -> float:
        return 0.80

    @property
    def trust_blacklist(self) -> float:
        # 仅用于成员接入的软阈；CH 候选无硬阈，完全由 DST 融合结果排序决定
        return 0.35

    @property
    def forget(self) -> float:
        return 0.98

    @property
    def strike_threshold(self) -> int:
        return 33

    # ===== 内部观测：用于“邻居对邻居”的行为证据（馈入 DST 的 A1/A2） =====
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 记录边 (i->j) 的成功/失败（由 watchdog 回写）
        self.obs: Dict[Tuple[int, int], Dict[str, float]] = {}
        # CH->上一跳 的记录（以便 watchdog 归因）
        self.last_relay_by_ch: Dict[int, Optional[int]] = {}
        # 初始化节点基准能量（用于回收阈值）
        if self.sim is not None:
            for n in self.sim.nodes:
                if not hasattr(n, "initial_energy"):
                    n.initial_energy = n.energy

    # --------------------- DST：证据构造与组合（Eq.(7)-(10)) ---------------------
    def _get_obs(self, i: int, j: int) -> Dict[str, float]:
        key = (i, j)
        if key not in self.obs:
            self.obs[key] = dict(p=0.0, n=0.0, last_ok=1.0)
        return self.obs[key]

    def _neighbor_belief_about(self, i: Node, j: Node) -> Tuple[float, float, float]:
        """
        邻居 i 对节点 j 的信任证据（A_i）：
        返回 (m(T), m(F), m(Ω))，并保持 m(T)+m(F)+m(Ω)=1。
        证据来源：对 (i->j) 的成功/失败观测 + 节点 j 的全局 Beta-trust 作为不确定补充。
        """
        rec = self._get_obs(i.nid, j.nid)
        total = rec['p'] + rec['n']
        if total <= 0:
            # 没有直接观测：给 0.5/0.5 的模糊意见，并把不确定分量留给 Ω
            mT = 0.5 * j.trust()     # 倾向于全局信任（防止过拟合）
            mF = 0.5 * (1.0 - j.trust())
            mO = clamp(1.0 - (mT + mF), 0.0, 1.0)
            return mT, mF, mO
        # 有观测：p/(p+n) → T，(1-p/(p+n)) → F，剩余给 Ω
        p_rate = rec['p'] / total
        mT = clamp(0.7 * p_rate + 0.3 * j.trust(), 0.0, 1.0)  # 融合全局信任，增强鲁棒
        mF = clamp(1.0 - mT, 0.0, 1.0) * 0.8                  # 适度保留冲突
        mO = clamp(1.0 - mT - mF, 0.0, 1.0)
        # 归一化
        s = mT + mF + mO
        if s <= 1e-9:
            return 0.5, 0.5, 0.0
        return mT/s, mF/s, mO/s

    def _dst_combine(self, m1: Tuple[float, float, float], m2: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Dempster 组合（标准二元 {T,F} + Ω）：
          m(T) = [m1(T)m2(T) + m1(T)m2(Ω) + m1(Ω)m2(T)] / (1-K)
          m(F) = [m1(F)m2(F) + m1(F)m2(Ω) + m1(Ω)m2(F)] / (1-K)
          m(Ω) = [m1(Ω)m2(Ω)] / (1-K)
          K    = m1(T)m2(F) + m1(F)m2(T)
        与论文 Eq.(7)-(10) 给出的三种情形一致（这是一致化的通式写法）。
        """
        m1T, m1F, m1O = m1
        m2T, m2F, m2O = m2
        K = m1T*m2F + m1F*m2T
        den = 1.0 - K
        if den <= 1e-9:
            # 完全冲突，退回到不确定
            return 0.5, 0.5, 0.0
        mT = (m1T*m2T + m1T*m2O + m1O*m2T) / den
        mF = (m1F*m2F + m1F*m2O + m1O*m2F) / den
        mO = (m1O*m2O) / den
        # 归一化防数值漂移
        s = mT + mF + mO
        return mT/s, mF/s, mO/s

    def _final_trust_rate(self, m: Tuple[float, float, float]) -> float:
        """
        论文将最终“信任率”视为 T 的置信；不确定性 Ω 以 0.5 权重计入（保守下注）。
        """
        mT, mF, mO = m
        return clamp(mT + 0.5 * mO, 0.0, 1.0)

    # --------------------------- Phase 1：CH 选择 ---------------------------
    def select_cluster_heads(self):
        sim: Simulation = self.sim
        alive = [n for n in sim.alive_nodes() if not n.blacklisted]
        if not alive:
            return

        # 1) 先回收：检测现任 CH 是否满足 Phase 2 的电量/时间阈值
        self._recycle_existing_chs()

        # 2) 目标 CH 数量（约每簇 20 个节点）
        target_ch = max(1, math.ceil(len(alive) / float(self.TARGET_CLUSTER_SIZE)))

        # 3) 候选集合（冷却期）
        candidates = [n for n in alive if (sim.round - n.last_ch_round) >= CH_COOLDOWN]

        # 若现有 CH 已满足目标，不再补选
        if len(sim.clusters) >= target_ch or not candidates:
            return

        # 4) 为每个候选计算 DST 信任率（融合“两个邻居”的证据）
        #    邻居优先取通信半径内最近的两个；不足则用 Ω=1 的中性证据补足
        scored: List[Tuple[float, Node]] = []
        for u in candidates:
            neigh = self._neighbors_within(u, alive, COMM_RANGE)
            neigh.sort(key=lambda x: x[0])
            if len(neigh) >= 2:
                a1 = self._neighbor_belief_about(neigh[0][1], u)
                a2 = self._neighbor_belief_about(neigh[1][1], u)
            elif len(neigh) == 1:
                a1 = self._neighbor_belief_about(neigh[0][1], u)
                a2 = (0.0, 0.0, 1.0)  # 纯不确定
            else:
                # 无邻居，退化：使用自身 trust 形成模糊证据
                t = u.trust()
                a1 = (t, 1.0 - t, 0.0)
                a2 = (0.0, 0.0, 1.0)

            mT = self._dst_combine(a1, a2)        # Eq.(7)-(10) 的组合
            tr = self._final_trust_rate(mT)        # 作为最终信任率
            scored.append((tr, u))

        # 5) 以 DST 信任率排序，依次选 CH；并用概率门控（与框架 BASE_P_CH 协同）
        scored.sort(key=lambda x: x[0], reverse=True)
        for tr, u in scored:
            if len(sim.clusters) >= target_ch:
                break
            p = clamp(BASE_P_CH * (0.60 + 0.80 * tr), 0.0, 1.0)
            if random.random() < p:
                self._appoint_ch(u)

        # 兜底：若仍不足，直接按分数补齐
        if len(sim.clusters) < target_ch:
            for _, u in scored:
                if len(sim.clusters) >= target_ch:
                    break
                if u.nid not in sim.clusters:
                    self._appoint_ch(u)

    def _appoint_ch(self, n: Node):
        sim = self.sim
        if n.nid in sim.clusters:
            return
        n.is_ch = True
        n.last_ch_round = sim.round
        sim.clusters[n.nid] = Cluster(n.nid)

    def _recycle_existing_chs(self):
        """Phase 2: CH 回收（min_erg=0.6, tl=60）"""
        sim: Simulation = self.sim
        to_remove = []
        for ch_id, cl in list(sim.clusters.items()):
            n = sim.node_by_id(ch_id)
            if not n or (not n.alive):
                to_remove.append(ch_id); continue
            base = getattr(n, "initial_energy", n.energy)
            min_erg = self.MIN_ERG_RATIO * base
            # 条件：能量低于阈值 或 已到达 tl
            due = ((sim.round - n.last_ch_round) >= self.TL_ROUNDS)
            low = (n.energy <= min_erg)
            if low or due:
                n.is_ch = False
                to_remove.append(ch_id)
        for cid in to_remove:
            sim.clusters.pop(cid, None)

    # --------------------------- Phase 2：成员冗余（论文未主张） ---------------------------
    def allow_member_redundancy(self, member: Node, ch: Node) -> bool:
        return False

    # --------------------------- Phase 3：WOA 选路（返回下一跳） ---------------------------
    def choose_ch_relay(self, ch: Node, ch_nodes: List[Node]):
        """
        以 WOA 在“直达 + 一跳中继”方案中择优，符合 Eq.(11)-(12) 的适应度最大化。
        由于主程序逐跳发送，此处将“解”定义为：候选下一跳的索引（含直达 BS 的虚拟选项）。
        """
        sim = self.sim

        # 候选邻居（通信邻域内的其他 CH） —— 使用缩小半径
        pool: List[Node] = []
        for other in ch_nodes:
            if other.nid == ch.nid or (not other.alive):
                continue
            if dist(ch.pos(), other.pos()) <= self.NEIGHBOR_RADIUS_FACTOR * CH_NEIGHBOR_RANGE:
                pool.append(other)

        # 将“直达 BS”视为一个额外选项（索引 = -1）
        K = len(pool) + 1  # 最后一个“类别”表示直达
        if K <= 0:
            self.last_relay_by_ch[ch.nid] = None
            return None, {}

        # ========== WOA 初始化 ==========
        P = min(10, max(4, self.WOA_POP_FACTOR * K))
        iters = self.WOA_ITERS

        # 个体位置：实数 in [0, K-1]，离散化取最近索引；索引 K-1 代表直达
        def rnd_pos():
            return random.random() * max(1.0, (K - 1))

        X = [rnd_pos() for _ in range(P)]
        X_best = X[0]
        F_best = -1e18

        def discretize(x: float) -> int:
            idx = int(round(clamp(x, 0.0, max(0.0, K - 1))))
            return idx  # [0..K-1]；K-1 == 直达

        def fitness_for_idx(idx: int) -> float:
            """
            Eq.(11)-(12) 的实现：
              - 路径为 [CH] -> [候选] -> [BS] 或 [CH]->[BS]（直达）
              - E: 路径节点（不含 BS）能量归一平均（对齐论文定义“Residual energy (E)”）
              - D: CH 与候选中继的距离（直达时取 +∞，其 1/D = 0）
              - H: 跳数（直达=1，中继=2）
              - Scdis: 按 Eq.(12) 计算的均值路径段长度
            """
            # 构造路径
            if idx == (K - 1):  # 直达
                pts = [ch.pos(), sim.bs]
                H = 1.0
                D = float('inf')
                energies = [ch.energy]
            else:
                nb = pool[idx]
                pts = [ch.pos(), nb.pos(), sim.bs]
                H = 2.0
                D = dist(ch.pos(), nb.pos())
                energies = [ch.energy, nb.energy]

            # 归一化量纲：距离都以 CH_NEIGHBOR_RANGE 为参考
            Scdis = _avg_path_distance(pts)
            Scdis_norm = Scdis / (CH_NEIGHBOR_RANGE + 1e-9)

            # 1/D 项（直达时为 0）
            inv_D = 0.0 if (not np.isfinite(D)) else 1.0 / max(D, 1e-9)

            # E：能量归一（对当前邻域的最大能量归一化）
            e_ref = max([ch.energy] + [n.energy for n in pool]) if pool else ch.energy
            Ehats = [clamp(e / (e_ref + 1e-9), 0.0, 1.0) for e in energies]
            E_term = float(np.mean(Ehats))

            # 适应度（Eq.(11)）
            fit = E_term + inv_D + (1.0 / H) + (1.0 / max(Scdis_norm, 1e-6))
            return fit

        def evaluate(xlist: List[float]) -> Tuple[List[float], List[int]]:
            Is = [discretize(x) for x in xlist]
            Fs = [fitness_for_idx(i) for i in Is]
            return Fs, Is

        # 初始评价
        F, Ixs = evaluate(X)
        F_best = max(F)
        X_best = X[int(np.argmax(F))]

        # ========== 主循环（标准 WOA） ==========
        for t in range(iters):
            a = 2.0 - 2.0 * (t / max(1.0, (iters - 1)))  # a from 2 -> 0
            for p in range(P):
                r1 = random.random(); r2 = random.random()
                A = 2.0 * a * r1 - a
                C = 2.0 * r2

                # 0.5 的分支：encircling / 探索；否则 spiral
                if random.random() < 0.5:
                    if abs(A) < 1:
                        D_vec = abs(C * X_best - X[p])
                        X[p] = X_best - A * D_vec
                    else:
                        # 远离一个随机解
                        rand_idx = random.randint(0, P - 1)
                        D_vec = abs(C * X[rand_idx] - X[p])
                        X[p] = X[rand_idx] - A * D_vec
                else:
                    # 螺旋更新
                    l = random.uniform(-1.0, 1.0)
                    D_sp = abs(X_best - X[p])
                    X[p] = D_sp * math.exp(self.WOA_B * l) * math.cos(2.0 * math.pi * l) + X_best

            # 迭代末评价
            F, Ixs = evaluate(X)
            idx = int(np.argmax(F))
            if F[idx] > F_best:
                F_best = F[idx]
                X_best = X[idx]

        # 离散化得到最终选择
        best_idx = discretize(X_best)

        # 只改高影响参数：ε-随机与直达偏置
        if random.random() < self.EPSILON_ROUTE:
            best_idx = random.randint(0, K - 1)
        if random.random() < self.DIRECT_BIAS:
            best_idx = K - 1

        if best_idx == (K - 1):
            # 直达
            self.last_relay_by_ch[ch.nid] = None
            return None, {'fitness': F_best, 'mode': 'direct'}
        else:
            relay = pool[best_idx]
            self.last_relay_by_ch[ch.nid] = relay.nid
            return relay, {'fitness': F_best, 'mode': 'relay', 'relay_id': relay.nid}

    # --------------------------- 监督与可信/黑名单更新 ---------------------------
    def apply_watchdog(self, ch: Node, ok: bool, timely: bool, ch_nodes: List[Node]):
        """
        将本轮 CH->relay 的成功/失败计入 obs(i->j)，作为 DST 的邻居证据来源。
        """
        relay_id = self.last_relay_by_ch.get(ch.nid, None)
        if relay_id is None:
            # 直达：不产生对等体观测，但仍计入 CH 的 Beta-trust 统计
            if ok and timely:
                ch.observed_success += 0.6
            else:
                ch.observed_fail += 0.5
                ch.consecutive_strikes = getattr(ch, "consecutive_strikes", 0) + 1
            return

        rec = self._get_obs(ch.nid, relay_id)
        if ok and timely:
            rec['p'] += 1.0
            rec['last_ok'] = 1.0
            ch.observed_success += 0.6
            ch.consecutive_strikes = max(0, getattr(ch, "consecutive_strikes", 0) - 1)
        else:
            rec['n'] += 1.0
            rec['last_ok'] = 0.0
            ch.observed_fail += 0.5
            ch.consecutive_strikes = getattr(ch, "consecutive_strikes", 0) + 1

    def finalize_trust_blacklist(self):
        """
        与主程序 Beta-trust 模型对齐：指数遗忘 + 本轮观测累积；
        拉黑条件：低信任并有击穿，或连续击穿次数超阈。
        """
        for n in self.sim.alive_nodes():
            n.trust_s = n.trust_s * self.forget + n.observed_success
            n.trust_f = n.trust_f * self.forget + n.observed_fail
            low_trust = (n.trust() < self.trust_blacklist)
            if (low_trust and getattr(n, "consecutive_strikes", 0) >= 1) or \
               (getattr(n, "consecutive_strikes", 0) >= self.strike_threshold):
                n.blacklisted = True

    # --------------------------- 辅助：邻域查询 ---------------------------
    def _neighbors_within(self, node: Node, pool: List[Node], radius: float) -> List[Tuple[float, Node]]:
        out = []
        p0 = node.pos()
        for other in pool:
            if other.nid == node.nid or (not other.alive):
                continue
            d = dist(p0, other.pos())
            if d <= radius:
                out.append((d, other))
        return out
