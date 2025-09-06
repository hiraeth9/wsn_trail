# -*- coding: utf-8 -*-
"""
TFSM-DPC（论文对齐版）
- 直接信任（带挥发 λ 与自适应惩罚 θ）：Eq.(4)–(7)
- 改进 DPC 过滤邻居推荐值（含两个人工基准，上/下界）：Eq.(12)–(14) 与 §4.2 步骤(1)-(9)
- 综合信任（有可信推荐用 Eq.(10)，否则 Eq.(11)），阈值判恶（CT_th=0.35）
参考：TFSM-DPC, Pervasive and Mobile Computing, 2024. :contentReference[oaicite:2]{index=2}
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

class TFSM_DPC(AlgorithmBase):
    """
    与 trail/core/wsn_core.py 框架兼容的 TFSM-DPC 落地版本。
    提供主程序期望的 5 个接口：
      - select_cluster_heads（保持与框架一致的能量/地理/概率选 CH）
      - allow_member_redundancy（成员侧冗余：关闭）
      - choose_ch_relay（依据综合信任 CT 选下一跳；若不达阈值则直达）
      - apply_watchdog（把当轮成功/失败喂给直接信任的 n_t/m_t）
      - finalize_trust_blacklist（回合末统一执行 λ 挥发更新与 Beta-trust 加权遗忘）
    """

    # ===== 论文参数/阈值（Table 2） =====
    CT_th: float = 0.35     # 综合信任阈值（判恶&选路门槛）  表2
    eta: float  = 0.50      # 综合信任权重：CT = (1-eta)*DT + eta*IT  Eq.(10)

    # 直接信任更新：挥发因子 λ（Eq.(6)）与自适应惩罚因子 θ（Eq.(7)）
    lam: float = 0.85       # 0<λ<1，越小越“遗忘”历史
    a2: float  = 8.0        # θ(AC)=1.5 - sigmoid(a2*AC) 的陡峭度（Eq.(7) 的工程化）

    # 改进 DPC：K=3，基准上/下界（§4.2 步骤(1)）
    K: int = 3
    bench_low: float  = 0.05   # “恶评”基准（bad-mouth）
    bench_high: float = 0.95   # “互捧”基准（collusion）
    dc_percentile: float = 0.02  # 截断距离 dc 取全对距离升序的 1%~2% 位置（本文取 2%）Eq.(12)

    # CH 选择（不属论文主体；与框架一致，能量/近邻/汇聚距离+概率抽样）
    alpha_e = 0.40  # energy
    beta_p  = 0.30  # proximity (邻域平均距离的反比)
    gamma_s = 0.30  # sink proximity

    # Beta-trust 黑名单逻辑（沿用框架风格）
    @property
    def name(self) -> str:
        return "TFSM-DPC-2024"

    @property
    def trust_warn(self) -> float:
        return 0.80

    @property
    def trust_blacklist(self) -> float:
        return 0.35

    @property
    def forget(self) -> float:
        return 0.98

    @property
    def strike_threshold(self) -> int:
        return 35

    # ================= 内部状态 =================
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 成对统计：α_t, β_t 及当轮增量 n_t, m_t（Eq.(6)）
        # key=(i,j)
        self.pairs: Dict[Tuple[int,int], Dict[str,float]] = {}
        # 记录每个 CH 上一次选中的中继，供 watchdog 归因
        self.last_relay_by_ch: Dict[int, Optional[int]] = {}

    # ---------- 工具：访问/创建成对统计 ----------
    def _rec(self, i: int, j: int) -> Dict[str,float]:
        key = (i, j)
        if key not in self.pairs:
            self.pairs[key] = dict(alpha=0.0, beta=0.0, n_cur=0.0, m_cur=0.0)
        return self.pairs[key]

    # ========== 直接信任（Eq.(4)-(7)) ==========
    def _theta(self, i: int, j: int) -> float:
        """自适应惩罚因子 θ(AC)=1.5 - sigmoid(a2*AC)，范围[1.0,1.5]（Eq.(7) 的等价形）。"""
        r = self._rec(i, j)
        nt, mt = r['n_cur'], r['m_cur']
        if (nt + mt) <= 1e-9:
            ac = 0.0
        else:
            ac = mt / (nt + mt)
        # sigmoid(x) = 1/(1+e^-x) → θ ∈ [1.0,1.5]
        return 1.5 - 1.0 / (1.0 + math.exp(-self.a2 * ac))

    def _dt_pair(self, i: int, j: int) -> float:
        """
        DT_t(i->j) = (α_t + 1) / (α_t + β_t + 2*θ)   （Eq.(5) + Eq.(7)）
        其中 α_t = λ·α_{t-1} + n_t；β_t = λ·β_{t-1} + m_t   （Eq.(6)）
        注意：这里“在线”视角，使用当前缓存的 α_t/β_t 与当轮增量计算出的等效值。
        """
        r = self._rec(i, j)
        a_eff = self.lam * r['alpha'] + r['n_cur']
        b_eff = self.lam * r['beta']  + r['m_cur']
        theta = self._theta(i, j)
        denom = a_eff + b_eff + 2.0 * theta
        return clamp((a_eff + 1.0) / max(1e-9, denom), 0.0, 1.0)

    # ========== 改进 DPC（Eq.(12)–(14)，§4.2） ==========
    def _dpc_filter_recom(self, i: Node, j: Node, commons: List[Node]) -> Tuple[List[Node], bool]:
        """
        输入：公共邻居 commons
        输出：可信集合 PBh（被用于 Eq.(8)-(9)），以及 attack_flag（若侦测到基准簇吸附真实点）
        步骤严格按 §4.2(1)-(9) 实现，K=3，引入两个人工基准（上/下界）。
        """
        if not commons:
            return [], False

        # 组装数据：x 为 1..q 的序号，y 为推荐的 DT_t(k->j)（§4.2）
        xs, ys, nodes = [], [], []
        for idx, k in enumerate(commons, 1):
            xs.append(float(idx))
            ys.append(self._dt_pair(k.nid, j.nid))
            nodes.append(k)

        # 注入两个人工基准（上/下界）：x 取 0 与 q+1（与真实点分隔）
        xs_bench = [0.0, float(len(xs) + 1)]
        ys_bench = [self.bench_low, self.bench_high]
        xs_all = xs_bench + xs
        ys_all = ys_bench + ys
        N = len(xs_all)

        # 计算两两距离
        P = np.stack([np.array(xs_all), np.array(ys_all)], axis=1)
        D = np.zeros((N, N), dtype=float)
        for a in range(N):
            for b in range(a+1, N):
                d = math.hypot(P[a,0]-P[b,0], P[a,1]-P[b,1])
                D[a,b] = D[b,a] = d

        # 截断距离 dc（全对距离升序第 2% 位置，Eq.(12)）
        upper = np.sort(D[np.triu_indices(N, 1)])
        if len(upper) == 0:
            return commons, False
        pos = max(0, int(round(self.dc_percentile * len(upper))) - 1)
        dc = max(upper[pos], 1e-6)

        # 局部密度 ρ_x（Eq.(12)）
        rho = np.zeros(N)
        for x in range(N):
            rho[x] = np.sum(np.exp(- (D[x]**2) / (dc**2)) ) - 1.0  # 去掉自身项

        # 比较距离 δ_x（Eq.(13)）
        delta = np.zeros(N)
        order = np.argsort(-rho)  # 按密度降序
        for rank, x in enumerate(order):
            if rank == 0:
                delta[x] = np.max(D[x])  # 最高密度点 δ = max 距离
            else:
                # 在更高密度集合中找最近距离
                higher = order[:rank]
                delta[x] = np.min(D[x, higher])

        # 决策值 φ_x（Eq.(14)）
        phi = rho * delta

        # 选 K 个中心
        K = min(self.K, N)
        centers = list(np.argsort(-phi)[:K])

        # 将其余点分配到最近的“更高密度邻居”的簇（经典 DPC 规则）
        labels = -np.ones(N, dtype=int)
        for ci, c in enumerate(centers):
            labels[c] = ci
        # 依密度降序把非中心点挂接到最近的高密度点所属簇
        for x in order:
            if labels[x] != -1:
                continue
            higher = [y for y in order if rho[y] > rho[x]]
            if not higher:
                # 没有更高密度点，直接找最近中心
                near_c = min(centers, key=lambda c: D[x,c])
                labels[x] = labels[near_c]
            else:
                y = min(higher, key=lambda h: D[x,h])
                labels[x] = labels[y]

        # 基准点索引：0=low，1=high
        lab_low  = labels[0]
        lab_high = labels[1]

        # 属于基准簇的“真实点”视为虚假推荐；其余簇为可信
        false_idx = set([k for k in range(2, N) if labels[k] in (lab_low, lab_high)])
        trusted_idx = [k for k in range(2, N) if k not in false_idx]

        # §4.2(9)：把虚假推荐作为“异常行为”计入 i->k 的 m_t
        for local_idx in false_idx:
            k_node = nodes[local_idx - 2]
            self._rec(i.nid, k_node.nid)['m_cur'] += 1.0

        # attack_flag：若存在任何真实点落在任一基准簇，视为遭受 bad-mouth 或/及 collusion
        attacked = (len(false_idx) > 0)

        PBh = [nodes[idx - 2] for idx in trusted_idx]
        return PBh, attacked

    # ========== 间接/综合信任（Eq.(8)-(11)) ==========
    def _it_pair(self, i: Node, j: Node, commons: List[Node]) -> Tuple[float, bool]:
        """
        计算 IT_t(i->j) 及 attack_flag：
          IT = Σ_{k∈PBh} ω_k * DT(i->k) * DT(k->j)          （Eq.(8)）
          ω_k = DT(i->k) / Σ DT(i->k)（k∈PBh）               （Eq.(9)）
        PBh 由改进 DPC 过滤得到；若出现攻击迹象，则后续 CT 退化为 DT（Eq.(11)）
        """
        PBh, attacked = self._dpc_filter_recom(i, j, commons)
        if not PBh:
            return 0.0, attacked
        dti_k = np.array([self._dt_pair(i.nid, k.nid) for k in PBh], dtype=float)
        if np.all(dti_k <= 1e-9):
            return 0.0, attacked
        w = dti_k / np.sum(dti_k)
        dt_kj = np.array([self._dt_pair(k.nid, j.nid) for k in PBh], dtype=float)
        IT = float(np.sum(w * dti_k * dt_kj))
        return clamp(IT, 0.0, 1.0), attacked

    def _ct_pair(self, i: Node, j: Node, ch_nodes: List[Node]) -> float:
        """CT = (1-η)DT + η·IT（若推荐可信）；否则 CT=DT（Eq.(10)/(11)）"""
        DT = self._dt_pair(i.nid, j.nid)
        # 公共邻居：两者皆在 CH 通信邻域（按 CH_NEIGHBOR_RANGE 近似）
        commons = []
        for k in ch_nodes:
            if k.nid in (i.nid, j.nid) or (not k.alive):
                continue
            if dist(i.pos(), k.pos()) <= CH_NEIGHBOR_RANGE and dist(j.pos(), k.pos()) <= CH_NEIGHBOR_RANGE:
                commons.append(k)
        IT, attacked = self._it_pair(i, j, commons)
        if attacked:
            return DT
        return clamp((1.0 - self.eta) * DT + self.eta * IT, 0.0, 1.0)

    # ================= 主程序钩子：簇首选择（与框架对齐，非论文重点） =================
    def select_cluster_heads(self):
        sim: Simulation = self.sim
        alive = [n for n in sim.alive_nodes() if not n.blacklisted]
        if not alive:
            return

        eligible = [n for n in alive if (sim.round - n.last_ch_round) >= CH_COOLDOWN]
        if not eligible:
            return

        e_max = max(n.energy for n in alive)
        d_sink_max = max(dist(n.pos(), sim.bs) for n in alive)

        # 概率抽样 + 保底 8%
        scored = []
        for n in eligible:
            e_term = clamp(n.energy / (e_max + 1e-9), 0.0, 1.0)
            # 邻近性：通信半径内邻居平均距离的反比
            neigh = self._neighbors_within(n, alive, COMM_RANGE)
            if neigh:
                davg = float(np.mean([d for d,_ in neigh]))
                prox = clamp(1.0 - davg / (COMM_RANGE + 1e-9), 0.0, 1.0)
            else:
                prox = 0.0
            # 到汇聚节点越近越好
            s_term = clamp(1.0 - dist(n.pos(), sim.bs) / (d_sink_max + 1e-9), 0.0, 1.0)

            score = self.alpha_e * e_term + self.beta_p * prox + self.gamma_s * s_term
            scored.append((score, n))

            p = clamp(BASE_P_CH * (0.60 + 0.80 * score), 0.0, 1.0)
            if random.random() < p:
                self._appoint_ch(n)

        need_total = max(1, int(0.08 * len(alive)))
        if len(sim.clusters) < need_total:
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, pick in scored[:(need_total - len(sim.clusters))]:
                self._appoint_ch(pick)

    def _appoint_ch(self, n: Node):
        sim = self.sim
        if n.nid in sim.clusters:
            return
        n.is_ch = True
        n.last_ch_round = sim.round
        sim.clusters[n.nid] = Cluster(n.nid)

    def allow_member_redundancy(self, member: Node, ch: Node) -> bool:
        return False

    # ================= 主程序钩子：CH→CH 选路（以综合信任 CT 最大化） =================
    def choose_ch_relay(self, ch: Node, ch_nodes: List[Node]):
        """
        在通信邻域内的其他 CH 中，择使 CT(ch->nb) 最大且 ≥ CT_th 的下一跳；
        若无达标候选，则返回 None（主程序会直达 BS）。
        """
        # 候选
        pool = []
        for other in ch_nodes:
            if other.nid == ch.nid or (not other.alive):
                continue
            if dist(ch.pos(), other.pos()) <= CH_NEIGHBOR_RANGE:
                pool.append(other)
        if not pool:
            self.last_relay_by_ch[ch.nid] = None
            return None, {}

        # 计算 CT 并选最佳
        scored = []
        for nb in pool:
            ct = self._ct_pair(ch, nb, ch_nodes)
            scored.append((ct, nb))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_ct, relay = scored[0]

        if best_ct >= self.CT_th:
            self.last_relay_by_ch[ch.nid] = relay.nid
            return relay, {'ct': best_ct}
        else:
            self.last_relay_by_ch[ch.nid] = None
            return None, {}

    # ================= 主程序钩子：Watchdog & 信任/黑名单 =================
    def apply_watchdog(self, ch: Node, ok: bool, timely: bool, ch_nodes: List[Node]):
        """
        将本轮 CH->relay 的结果计入 n_t/m_t（直接信任的“当轮观测”），并同步节点级统计。
        """
        relay_id = self.last_relay_by_ch.get(ch.nid, None)
        if relay_id is not None:
            rec = self._rec(ch.nid, relay_id)
            if ok and timely:
                rec['n_cur'] += 1.0
                ch.observed_success += 0.6
                ch.consecutive_strikes = max(0, getattr(ch, "consecutive_strikes", 0) - 1)
            else:
                rec['m_cur'] += 1.0
                ch.observed_fail += 0.5
                ch.consecutive_strikes = getattr(ch, "consecutive_strikes", 0) + 1
        else:
            # 直达：只更新节点自身观测
            if ok and timely:
                ch.observed_success += 0.6
            else:
                ch.observed_fail += 0.5
                ch.consecutive_strikes = getattr(ch, "consecutive_strikes", 0) + 1

    def finalize_trust_blacklist(self):
        """
        回合末处理：
        1) 对所有成对统计执行 α_t=λ·α + n_t，β_t=λ·β + m_t（Eq.(6)），并清零当轮计数；
        2) 节点级 Beta-trust：指数遗忘并判断黑名单（与框架一致）。
        """
        # 1) α/β 挥发更新
        for key, r in self.pairs.items():
            r['alpha'] = self.lam * r['alpha'] + r['n_cur']
            r['beta']  = self.lam * r['beta']  + r['m_cur']
            r['n_cur'] = 0.0
            r['m_cur'] = 0.0

        # 2) 节点级 Beta-trust/黑名单
        for n in self.sim.alive_nodes():
            n.trust_s = n.trust_s * self.forget + n.observed_success
            n.trust_f = n.trust_f * self.forget + n.observed_fail
            low_trust = (n.trust() < self.trust_blacklist)
            if (low_trust and getattr(n, "consecutive_strikes", 0) >= 1) or \
               (getattr(n, "consecutive_strikes", 0) >= self.strike_threshold):
                n.blacklisted = True

    # ================= 辅助：邻居查询 =================
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
