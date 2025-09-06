# -*- coding: utf-8 -*-
from typing import List
import random
import numpy as np

from core.wsn_core import (
    AlgorithmBase, Simulation, Node, Cluster,
    dist, e_tx, e_rx, clamp,
    BASE_P_CH, CH_COOLDOWN,
    COMM_RANGE, CH_NEIGHBOR_RANGE,
    DATA_PACKET_BITS
)

class SHEER(AlgorithmBase):
    """
    SHEER（论文对齐版）
    - CH 适应度（式(6)）:
        CS = α·Eres/Emax + β·(1 - D_prox/D_max) + γ·(1 - D_sink/D_maxSink) + δ·T_node
    - CH→CH 中继（式(8)）:
        λ1·Eres/Emax + λ2·(1 - D_CH-CH/D_max) + λ3·T_CH
    - 自愈（式(7)）:
        触发条件：能量≤阈值 或 本轮传输非及时达（timely=False）
    """

    # ======== 论文权重（可按实验微调） ========
    # 式(6)权重
    alpha_e = 0.35  # energy
    beta_p  = 0.25  # proximity / connectivity
    gamma_s = 0.20  # sink distance (closer is better)
    delta_t = 0.20  # trust

    # 式(8)权重
    lam_e = 0.40
    lam_d = 0.25
    lam_t = 0.35

    # 论文信任阈值（T >= 0.7 才具备 CH 候选资格）
    ch_trust_threshold = 0.70

    # 自愈能量阈值：E_res <= energy_fail_ratio *（全网能量中位数）视为能量击穿
    energy_fail_ratio = 0.10

    # 两跳能耗门槛：两跳总能耗 <= eta_twohop * 直达能耗 才启用中继
    eta_twohop = 1.08

    # ======== 主程序对接的参数/阈值 ========
    @property
    def name(self) -> str:
        return "SHEER-2025"

    @property
    def trust_warn(self) -> float:
        # 内部保守提示阈值（不影响 CH 资格硬阈值 0.7）
        return 0.80

    @property
    def trust_blacklist(self) -> float:
        # 成员接入的最低信任阈值（主程序 assign_members 会用到）
        return 0.35

    @property
    def forget(self) -> float:
        # 信任的指数遗忘系数
        return 0.98

    @property
    def strike_threshold(self) -> int:
        # 连续击穿达到该次数强制拉黑
        return 35

    # ======== 内部工具 ========
    def _neighbors_within(self, node: Node, alive: List[Node], radius: float):
        out = []
        p0 = node.pos()
        for other in alive:
            if other.nid == node.nid or (not other.alive):
                continue
            d = dist(p0, other.pos())
            if d <= radius:
                out.append((d, other))
        return out

    def _compute_proximity_term(self, node: Node, alive: List[Node]) -> float:
        """
        D_prox 取通信半径内邻居的平均距离；无邻居按最差 (= D_max) 处理
        返回项：(1 - D_prox / D_max) ∈ [0, 1]
        """
        neigh = self._neighbors_within(node, alive, COMM_RANGE)
        if not neigh:
            return 0.0
        avg_d = float(np.mean([d for d, _ in neigh]))
        return clamp(1.0 - (avg_d / (COMM_RANGE + 1e-9)), 0.0, 1.0)

    def _compute_sink_term(self, node: Node, d_sink_max: float) -> float:
        """(1 - D_sink / D_maxSink)"""
        d = dist(node.pos(), self.sim.bs)
        return clamp(1.0 - d / (d_sink_max + 1e-9), 0.0, 1.0)

    def _compute_trust_node(self, node: Node) -> float:
        """
        论文式(4)(5)：T = ω1·T_direct(PDR) + ω2·T_indirect(C_hist), ω1+ω2=1
        框架中 node.trust() 已是 Beta 先验 + 成功/失败观测的估计；
        这里再融合成员直达/中继两类历史值（若有）以贴近“直接/间接信任”。
        """
        base = node.trust()
        if hasattr(node, "dir_val") and hasattr(node, "rly_val"):
            hist = 0.5 * float(node.dir_val) + 0.5 * float(node.rly_val)
            return clamp(0.6 * base + 0.4 * hist, 0.0, 1.0)
        return base

    def _fitness_ch(self, node: Node, alive: List[Node], e_max: float, d_sink_max: float) -> float:
        e_term = clamp(node.energy / (e_max + 1e-9), 0.0, 1.0)
        p_term = self._compute_proximity_term(node, alive)
        s_term = self._compute_sink_term(node, d_sink_max)
        t_term = self._compute_trust_node(node)
        return (
            self.alpha_e * e_term +
            self.beta_p  * p_term +
            self.gamma_s * s_term +
            self.delta_t * t_term
        )

    # ======== 与主程序对接的必要接口 ========
    def select_cluster_heads(self):
        sim: Simulation = self.sim
        alive = [n for n in sim.alive_nodes() if not n.blacklisted]
        if not alive:
            return

        e_max = max(n.energy for n in alive)
        d_sink_max = max(dist(n.pos(), sim.bs) for n in alive)

        # 1) 概率抽样：CS 归一映射为抽样概率，与 BASE_P_CH 协同
        scored = []
        for n in alive:
            if (sim.round - n.last_ch_round) < CH_COOLDOWN:
                continue
            # 论文硬阈值：T >= 0.7 才有资格参选 CH
            if self._compute_trust_node(n) < self.ch_trust_threshold:
                continue

            cs = self._fitness_ch(n, alive, e_max, d_sink_max)  # Eq.(6)
            scored.append((cs, n))

            p = clamp(BASE_P_CH * (0.60 + 0.80 * cs), 0.0, 1.0)
            if random.random() < p:
                n.is_ch = True
                n.last_ch_round = sim.round
                sim.clusters[n.nid] = Cluster(n.nid)

        # 2) 保底：按 CS 高低补齐至期望下限（例如 ≥8% 活跃节点为 CH）
        need_total = max(1, int(0.08 * len(alive)))
        if len(sim.clusters) < need_total:
            if not scored:
                # 极端场景：放宽一次资格，全部重打分
                for n in alive:
                    scored.append((self._fitness_ch(n, alive, e_max, d_sink_max), n))
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, pick in scored[: (need_total - len(sim.clusters))]:
                if pick.nid not in sim.clusters:
                    pick.is_ch = True
                    pick.last_ch_round = sim.round
                    sim.clusters[pick.nid] = Cluster(pick.nid)

    def allow_member_redundancy(self, member: Node, ch: Node) -> bool:
        # 论文未主张成员侧冗余多路径，保持关闭以贴合能耗模型
        return False

    def choose_ch_relay(self, ch: Node, ch_nodes: List[Node]):
        """
        簇首→簇首中继（式(8)）：
        - 在邻近 CH 集合内，按 (能量/距离/信任) 加权计算通信适应度；
        - 仅当两跳总能耗不明显劣于直达（cost_twohop <= eta_twohop * cost_direct）才采用中继。
        """
        sim = self.sim

        # 直达代价（与主程序能耗模型一致）
        d_bs = dist(ch.pos(), sim.bs)
        cost_direct = e_tx(DATA_PACKET_BITS, d_bs)

        # 候选邻居
        pool = [x for x in ch_nodes if x.alive and x.nid != ch.nid]
        if not pool:
            return None, {}

        cands = []
        e_max = max(x.energy for x in pool) if pool else ch.energy
        for other in pool:
            d1 = dist(ch.pos(), other.pos())
            if d1 > CH_NEIGHBOR_RANGE:
                continue

            # 式(8)三项
            e_term = clamp(other.energy / (e_max + 1e-9), 0.0, 1.0)
            d_term = clamp(1.0 - d1 / (CH_NEIGHBOR_RANGE + 1e-9), 0.0, 1.0)
            t_term = self._compute_trust_node(other)

            fit = self.lam_e * e_term + self.lam_d * d_term + self.lam_t * t_term  # Eq.(8)

            # 两跳真实能耗
            d2 = dist(other.pos(), sim.bs)
            cost_twohop = (
                e_tx(DATA_PACKET_BITS, d1) +
                e_rx(DATA_PACKET_BITS) +
                e_tx(DATA_PACKET_BITS, d2)
            )
            cands.append((fit, other, d1, d2, cost_twohop))

        if not cands:
            return None, {}

        cands.sort(key=lambda x: x[0], reverse=True)
        best_fit, relay, d1, d2, cost2 = cands[0]

        use_relay = (
            relay is not None
            and self._compute_trust_node(relay) >= self.ch_trust_threshold
            and cost2 <= self.eta_twohop * cost_direct
        )

        return (relay if use_relay else None), (
            {'d1': d1, 'd2': d2, 'cost2': cost2, 'cost1': cost_direct, 'fit': best_fit}
            if use_relay else {}
        )

    def apply_watchdog(self, ch: Node, ok: bool, timely: bool, ch_nodes: List[Node]):
        """
        自愈监测（式(7)）：
        - 能量低（相对全网能量中位数的固定比例阈值近似 E_threshold）或本轮传输非及时达 → 记一次击穿；
        - 正常及时达则衰减可疑度。具体拉黑与信任更新在 finalize 中统一处理。
        """
        alive = self.sim.alive_nodes()
        med_e = float(np.median([n.energy for n in alive])) if alive else 0.0
        energy_breach = (ch.energy <= max(1e-6, self.energy_fail_ratio * med_e))
        time_breach = (not timely)

        if (not ok) or energy_breach or time_breach:
            ch.observed_fail += 0.5
            ch.suspicion = min(1.0, ch.suspicion + (0.20 if energy_breach else 0.12))
            ch.consecutive_strikes += 1
        else:
            ch.observed_success += 0.6
            ch.suspicion = max(0.0, ch.suspicion * 0.85)
            ch.consecutive_strikes = max(0, ch.consecutive_strikes - 1)

    def finalize_trust_blacklist(self):
        """
        信任更新 + 黑名单策略：
        - 指数遗忘 + 本轮观测成功/失败叠加 + 可疑度轻惩罚；
        - 拉黑条件：低信任且至少 1 次击穿，或 连续击穿次数 ≥ strike_threshold。
        """
        for n in self.sim.alive_nodes():
            n.trust_s = n.trust_s * self.forget + n.observed_success
            n.trust_f = n.trust_f * self.forget + n.observed_fail + 0.15 * n.suspicion

            low_trust = (n.trust() < self.trust_blacklist)
            if (low_trust and n.consecutive_strikes >= 1) or (n.consecutive_strikes >= self.strike_threshold):
                n.blacklisted = True
