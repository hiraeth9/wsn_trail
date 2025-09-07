
# -*- coding: utf-8 -*-
from typing import List
import random, numpy as np
from core.wsn_core import AlgorithmBase, Simulation, Node, Cluster, dist, clamp, \
    BASE_P_CH, CH_COOLDOWN, COMM_RANGE, CH_NEIGHBOR_RANGE, \
    e_tx, e_rx, DATA_PACKET_BITS

class TRAIL(AlgorithmBase):
    @property
    def name(self): return "TRAIL (ours)"
    @property
    def trust_warn(self): return 0.68
    @property
    def trust_blacklist(self): return 0.42

    @property
    def forget(self):
        return 0.94

    @property
    def strike_threshold(self):
        return 4

    def __init__(self, sim: 'Simulation'):
        super().__init__(sim)
        # —— 探索/冗余/能耗-可靠性策略参数 ——
        # —— 探索/冗余/能耗-可靠性策略参数（更稳健） ——
        self.rt_ratio = 0.30
        self.epsilon = 0.16  # 更快收敛，少随机抖动
        self.eps_decay = 0.992
        self.min_epsilon = 0.02
        self.alpha = 1.02  # 初始更保守
        self.queue_pen = 0.030  # 队列拥塞惩罚略增
        self.trust_pen = 0.060  # 对低信任更敏感
        self.rel_w = 0.48  # 提高“可靠性记忆”权重
        self.mode_by_ch = {}

        # —— 自适应边界稍收紧：高恶意时更保守 ——
        self.alpha_min, self.alpha_max = 0.98, 1.06
        self.alpha_step = 0.01

        self._rel_s = self._rel_t = 0
        self._dir_s = self._dir_t = 0

        # 冗余强度边界适度放宽（供自适应上调）
        self.rt_min, self.rt_max = 0.10, 0.45

    def select_cluster_heads(self):
        sim = self.sim
        alive = [n for n in sim.alive_nodes() if not n.blacklisted]
        if not alive: return
        energies = [n.energy for n in alive];
        e_min, e_max = min(energies), max(energies)
        dists = [dist(n.pos(), sim.bs) for n in alive];
        d_min, d_max = min(dists), max(dists)
        q_levels = [n.last_queue_level for n in alive];
        q_min, q_max = min(q_levels), max(q_levels)

        for n in alive:
            if (sim.round - n.last_ch_round) < CH_COOLDOWN:
                continue
            e = (n.energy - e_min) / (e_max - e_min + 1e-9)
            t = n.trust()
            q = 1.0 - (n.last_queue_level - q_min) / (q_max - q_min + 1e-9)
            b = 1.0 - (dist(n.pos(), sim.bs) - d_min) / (d_max - d_min + 1e-9)
            s_pen = 1.0 - max(0.0, min(1.0, 1.0 - n.suspicion))

            # 能量/信任权重更高，怀疑惩罚降权；提高抽样基线避免稀疏
            score = 0.40 * e + 0.30 * (t - 0.03 * s_pen) + 0.18 * q + 0.08 * b
            p = clamp(BASE_P_CH * (0.70 + score), 0.0, 1.0)
            if random.random() < p:
                n.is_ch = True;
                n.last_ch_round = sim.round
                sim.clusters[n.nid] = Cluster(n.nid)

        # —— 自适应：CH 至少覆盖 8% 节点，平均队列越高上限越高（最多到 12%）
        avg_q = float(np.mean([n.last_queue_level for n in alive])) if alive else 0.0
        target_ratio = 0.08 + clamp((avg_q - 1.0) * 0.01, 0.0, 0.04)  # 8% ~ 12%
        need_total = max(1, int(target_ratio * len(alive)))
        if len(sim.clusters) < need_total:
            scored = []
            for nn in [x for x in alive if not x.is_ch]:
                ee = (nn.energy - e_min) / (e_max - e_min + 1e-9)
                tt = nn.trust()
                qq = 1.0 - (nn.last_queue_level - q_min) / (q_max - q_min + 1e-9)
                bb = 1.0 - (dist(nn.pos(), sim.bs) - d_min) / (d_max - d_min + 1e-9)
                ss = 1.0 - max(0.0, min(1.0, 1.0 - nn.suspicion))
                sc = 0.44 * ee + 0.26 * (tt - 0.10 * ss) + 0.15 * qq + 0.15 * bb
                scored.append((sc, nn))
            scored.sort(key=lambda x: x[0], reverse=True)
            need = need_total - len(sim.clusters)
            for _, pick in scored[:max(0, need)]:
                pick.is_ch = True
                pick.last_ch_round = sim.round
                sim.clusters[pick.nid] = Cluster(pick.nid)

    def allow_member_redundancy(self, member: Node, ch: Node) -> bool:
        alive = self.sim.alive_nodes()
        if not (self.trust_blacklist <= ch.trust() < self.trust_warn):
            return False
        if not alive:
            return False
        import numpy as _np
        median_e = float(_np.median([n.energy for n in alive]))
        if member.energy < median_e:  # 低电成员不做冗余，保寿命
            return False

        # 按风险自适应提高冗余强度（但留有上限）
        risk = float(
            _np.mean([1.0 if (getattr(n, "blacklisted", False) or n.suspicion >= 0.60) else 0.0 for n in alive]))
        boost = 1.0 + 0.5 * risk  # 风险 50% 时，冗余概率提升 25%
        prob = min(0.95, self.rt_ratio * (1.0 - ch.trust()) * boost)
        return (random.random() < prob)

    def choose_ch_relay(self, ch: Node, ch_nodes: list):
        """
        风险自适应中继选择：
        - 根据全网“黑名单/高可疑占比”计算 risk（0~1），动态抬升中继信任门槛 t_floor；
        - 评分：两跳能耗 * 拥塞/信任惩罚(有下限) * 可靠性记忆奖励(有下限) * 高信任奖励；
        - 探索也有最低信任底线，避免乱跳到低信任中继。
        """
        sim = self.sim

        # 直达成本（做对比用）
        d_bs = dist(ch.pos(), sim.bs)
        cost_direct = e_tx(DATA_PACKET_BITS, d_bs)

        # —— 估计环境“敌意度”：黑名单或高可疑（>=0.60）的比例 ——
        alive_all = list(sim.alive_nodes())
        risk = 0.0
        if alive_all:
            risk = float(np.mean([1.0 if (getattr(n, "blacklisted", False) or n.suspicion >= 0.60) else 0.0
                                  for n in alive_all]))

        # —— 自适应信任门槛：恶意越多，门槛越高（至少高过黑名单阈值一点点） ——
        t_floor = max(self.trust_blacklist + 0.08, 0.50 + 0.20 * risk)

        # 能量中位数：低电量节点不当中继，避免后期失稳
        e_med = float(np.median([n.energy for n in alive_all])) if alive_all else 0.0

        # —— 形成候选池（先做硬过滤，再算分） ——
        cand_pool = []
        for other in ch_nodes:
            if other.nid == ch.nid or (not other.alive):
                continue

            d1 = dist(ch.pos(), other.pos())
            if d1 > CH_NEIGHBOR_RANGE:
                continue

            # 信任硬门槛 + 能量硬门槛
            if other.trust() < t_floor:
                continue
            if e_med > 0.0 and other.energy < 0.90 * e_med:
                continue

            d2 = dist(other.pos(), sim.bs)

            # 两跳真实能耗（Tx + Rx + Tx）
            two_hop_cost = e_tx(DATA_PACKET_BITS, d1) + e_rx(DATA_PACKET_BITS) + e_tx(DATA_PACKET_BITS, d2)

            # 可靠性记忆（你代码里已有的 EWMA 指标：直传稳定性代理，范围[0,1]）
            rel = other.dir_val

            # —— 惩罚/奖励，带下限钳制（防止极端值套利） ——
            penalty = 1.0 + self.queue_pen * other.queue_level + self.trust_pen * (1.0 - other.trust())
            penalty = max(1.05, penalty)  # 惩罚下限 1.05
            reward = 1.0 - self.rel_w * rel
            reward = max(0.62, reward)  # 奖励下限（最多降到 0.62 倍）
            trust_bonus = 1.0 - 0.20 * (1.0 - other.trust())

            score = two_hop_cost * penalty * reward * trust_bonus
            cand_pool.append((score, other, d1, d2, two_hop_cost))

        # 没有合格候选 → 走直达
        if not cand_pool:
            self.mode_by_ch[ch.nid] = 'direct'
            return None, {}

        # 取评分最低的候选
        cand_pool.sort(key=lambda x: x[0])
        best = cand_pool[0]
        relay = best[1]
        d1, d2 = best[2], best[3]
        best_cost = best[4]

        # —— ε-贪心探索 + 拥塞门槛（沿用你原逻辑） ——
        explore = (random.random() < self.epsilon)
        net_avg_q = float(np.mean([n.last_queue_level for n in alive_all])) if alive_all else 0.0
        qlimit = 1 if net_avg_q > 1.5 else 2

        if explore:
            # 探索也要有底线：只在“信任 ≥ t_floor”的候选里随机
            use_relay = (relay.trust() >= t_floor) and (random.random() < 0.5)
        else:
            use_relay = (best_cost + 1e-12) < (self.alpha * cost_direct) and \
                        (relay.queue_level <= ch.queue_level + qlimit) and (relay.energy > 0.08) and \
                        (relay.trust() >= t_floor)

        # 衰减探索率 + 记录模式
        self.epsilon = max(self.min_epsilon, self.epsilon * self.eps_decay)
        self.mode_by_ch[ch.nid] = ('relay' if use_relay else 'direct')

        return (relay if use_relay else None), (
            {'d1': d1, 'd2': d2, 'cost2': best_cost, 'cost1': cost_direct} if use_relay else {}
        )

    def apply_watchdog(self, ch: Node, ok: bool, timely: bool, ch_nodes: List[Node]):
        # 选最近的3个观察者（用于“目击”但不重复计数）
        neigh = []
        for other in ch_nodes:
            if other.nid == ch.nid or (not other.alive): continue
            neigh.append((dist(ch.pos(), other.pos()), other))
        neigh.sort(key=lambda x: x[0])
        watchers = [p[1] for p in neigh[:3]]

        mode = self.mode_by_ch.get(ch.nid, 'direct')
        # 只记录一次“全局成败”（避免被多个观察者重复算）
        if mode == 'direct':
            self._dir_t += 1
            if ok: self._dir_s += 1
        else:
            self._rel_t += 1
            if ok: self._rel_s += 1

        for _ in watchers:
            if ok and timely:
                # 成功：更快“去可疑化”，增强对应路径的 EWMA
                ch.suspicion = max(0.0, ch.suspicion - 0.06)
                ch.observed_success += 0.45
                if mode == 'direct':
                    ch.dir_val = 0.88 * ch.dir_val + 0.12 * 1.0
                    ch.dir_cnt += 1
                else:
                    ch.rly_val = 0.88 * ch.rly_val + 0.12 * 1.0
                    ch.rly_cnt += 1
            else:
                # 失败：relay 失败更重惩罚；直传稍轻，避免全局过度悲观
                if mode == 'relay':
                    if random.random() < 0.72:
                        ch.observed_fail += 0.70
                        ch.suspicion = min(1.0, ch.suspicion + 0.22)
                        ch.consecutive_strikes += 1
                    ch.rly_val *= 0.90
                    ch.dir_val *= 0.95
                else:
                    if random.random() < 0.65:
                        ch.observed_fail += 0.55
                        ch.suspicion = min(1.0, ch.suspicion + 0.16)
                    ch.dir_val *= 0.94
                    ch.rly_val *= 0.95

    def finalize_trust_blacklist(self):
        sim = self.sim
        for n in sim.alive_nodes():
            n.trust_s = n.trust_s * self.forget + n.observed_success
            n.trust_f = n.trust_f * self.forget + n.observed_fail + 0.20 * n.suspicion
            # 需要“低信任+至少一次击穿”，或“连续击穿超阈”，或“高可疑(≥0.85)+出现过击穿”
            if (n.trust() < self.trust_blacklist and n.consecutive_strikes >= 1) or \
                    (n.consecutive_strikes >= self.strike_threshold) or \
                    (n.suspicion >= 0.85 and n.consecutive_strikes >= 1):
                n.blacklisted = True

        # 基于最近的 direct/relay PDR 微调两跳容忍度与冗余强度（避免抖动）
        alive_now = sim.alive_nodes()
        tot = self._rel_t + self._dir_t
        if tot >= max(10, int(0.2 * len(alive_now) + 1)):
            rel_pdr = (self._rel_s / max(1, self._rel_t))
            dir_pdr = (self._dir_s / max(1, self._dir_t))
            if rel_pdr > dir_pdr + 0.03:
                self.alpha = min(self.alpha_max, self.alpha + self.alpha_step)
                self.rt_ratio = min(self.rt_max, self.rt_ratio * 1.05)
            elif dir_pdr > rel_pdr + 0.03:
                self.alpha = max(self.alpha_min, self.alpha - self.alpha_step)
                self.rt_ratio = max(self.rt_min, self.rt_ratio * 0.95)
            # 指数遗忘样本，保持灵敏但不震荡
            self._rel_s *= 0.5;
            self._rel_t *= 0.5
            self._dir_s *= 0.5;
            self._dir_t *= 0.5
        # —— 保留你现有的 PDR 自适应逻辑（无需改动） ——

        # —— 新增：按风险调整“上界”，恶意多时更保守 ——
        alive_now = sim.alive_nodes()
        risk = 0.0
        if alive_now:
            risk = float(
                np.mean([1.0 if (getattr(n, "blacklisted", False) or n.suspicion >= 0.60) else 0.0 for n in alive_now]))

        if risk >= 0.30:
            # 恶意比例偏高：中继阈值上界更紧，允许更强冗余
            self.alpha_max = 1.04
            self.rt_max = 0.50
        else:
            # 风险较低：恢复更宽松的上界
            self.alpha_max = 1.06
            self.rt_max = max(self.rt_max, 0.45)  # 不降低下来的好趋势


