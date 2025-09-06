
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
        self.rt_ratio = 0.30  # 提高冗余比率以增加可靠性
        self.epsilon = 0.18  # 降低初始探索率，减少随机性
        self.eps_decay = 0.995  # 稍微加快探索衰减
        self.min_epsilon = 0.03
        self.alpha = 1.05  # 放宽两跳能耗阈值，提高中继使用率
        self.queue_pen = 0.025  # 降低队列惩罚，鼓励使用中继
        self.trust_pen = 0.035  # 降低信任惩罚
        self.rel_w = 0.40  # 提高可靠性记忆权重
        self.mode_by_ch = {}  # 记录上一轮CH采用 direct/relay
        # —— 自适应调参用的全局计数 & 边界 ——
        self.alpha_min, self.alpha_max = 0.95, 1.08
        self.alpha_step = 0.01
        # 统计 direct/relay 在“看门狗判定”下的成功/总数，用于调 alpha / 冗余强度
        self._rel_s = self._rel_t = 0
        self._dir_s = self._dir_t = 0
        # 冗余强度边界，供自适应调整
        self.rt_min, self.rt_max = 0.10, 0.35


    def select_cluster_heads(self):
        sim=self.sim
        alive=[n for n in sim.alive_nodes() if not n.blacklisted]
        if not alive: return
        energies=[n.energy for n in alive]; e_min, e_max=min(energies), max(energies)
        dists=[dist(n.pos(), sim.bs) for n in alive]; d_min, d_max=min(dists), max(dists)
        q_levels=[n.last_queue_level for n in alive]; q_min, q_max=min(q_levels), max(q_levels)
        for n in alive:
            if (sim.round - n.last_ch_round) < CH_COOLDOWN: continue
            e = (n.energy - e_min) / (e_max - e_min + 1e-9)
            t = n.trust()
            q = 1.0 - (n.last_queue_level - q_min) / (q_max - q_min + 1e-9)
            b = 1.0 - (dist(n.pos(), sim.bs) - d_min) / (d_max - d_min + 1e-9)
            s_pen = 1.0 - max(0.0, min(1.0, 1.0 - n.suspicion))
            # 优化权重：提高信任权重，降低怀疑惩罚，增加队列和位置权重
            score = 0.40 * e + 0.30 * (t - 0.03 * s_pen) + 0.18 * q + 0.8 * b
            p = clamp(BASE_P_CH * (0.70 + score), 0.0, 1.0)
            if random.random()<p:
                n.is_ch=True; n.last_ch_round=sim.round
                sim.clusters[n.nid]=Cluster(n.nid)
        # —— 自适应 CH 比例：基础8%，平均队列每+1层 → 目标提升+1%，封顶+12% ——

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

    def allow_member_redundancy(self, member:Node, ch:Node)->bool:
        alive = self.sim.alive_nodes()
        if not (self.trust_blacklist <= ch.trust() < self.trust_warn):
            return False
        if not alive:
            return False
        import numpy as _np
        median_e = float(_np.median([n.energy for n in alive]))
        if member.energy < median_e:  # 低电成员不做冗余，保寿命
            return False
        prob = self.rt_ratio * (1.0 - ch.trust())  # CH 越可疑，冗余概率越高
        return (random.random() < prob)

    def choose_ch_relay(self, ch: Node, ch_nodes: List[Node]):
        sim = self.sim
        d_bs = dist(ch.pos(), sim.bs)
        cost_direct = e_tx(DATA_PACKET_BITS, d_bs)
        cands = []
        for other in ch_nodes:
            if other.nid == ch.nid or (not other.alive) or other.trust() < self.trust_blacklist:
                continue
            d1 = dist(ch.pos(), other.pos())
            if d1 > CH_NEIGHBOR_RANGE:
                continue
            d2 = dist(other.pos(), sim.bs)
            two_hop_cost = e_tx(DATA_PACKET_BITS, d1) + e_rx(DATA_PACKET_BITS) + e_tx(DATA_PACKET_BITS, d2)
            # 优化中继选择：降低队列惩罚，提高信任和可靠性权重
            rel = other.dir_val  # [0,1] 的EWMA
            penalty = (1.0 + self.queue_pen * other.queue_level + self.trust_pen * (1.0 - other.trust()))
            reward = max(0.65, (1.0 - self.rel_w * rel))  # 最多降低 35% 的“有效成本”
            # 额外奖励高信任中继
            trust_bonus = 1.0 - 0.15 * (1.0 - other.trust())
            score = two_hop_cost * penalty * reward * trust_bonus
            cands.append((score, other, d1, d2, two_hop_cost))
        if not cands:
            self.mode_by_ch[ch.nid] = 'direct'
            return None, {}
        cands.sort(key=lambda x: x[0])
        best = cands[0];
        relay = best[1];
        d1, b_d2 = best[2], best[3];
        best_cost = best[4]
        # 采用ε-贪心探索
        explore = (random.random() < self.epsilon)
        # 拥塞感知：网络平均队列高时，要求中继比CH更“空”（+1）；否则放宽到+2
        net_avg_q = float(np.mean([n.last_queue_level for n in sim.alive_nodes()])) if sim.alive_nodes() else 0.0
        qlimit = 1 if net_avg_q > 1.5 else 2
        if explore:
            use_relay = (random.random() < 0.5)
        else:
            use_relay = (best_cost + 1e-12) < (self.alpha * cost_direct) and \
                        (relay.queue_level <= ch.queue_level + qlimit) and (relay.energy > 0.08)

        # 衰减探索率
        self.epsilon = max(self.min_epsilon, self.epsilon * self.eps_decay)
        self.mode_by_ch[ch.nid] = ('relay' if use_relay else 'direct')
        return (relay if use_relay else None), (
            {'d1': d1, 'd2': b_d2, 'cost2': best_cost, 'cost1': cost_direct} if use_relay else {})

    def apply_watchdog(self, ch:Node, ok:bool, timely:bool, ch_nodes:List[Node]):
        neigh=[]
        for other in ch_nodes:
            if other.nid==ch.nid or (not other.alive): continue
            neigh.append((dist(ch.pos(), other.pos()), other))
        neigh.sort(key=lambda x: x[0]);
        watchers = [p[1] for p in neigh[:3]]  # 多一个观察者更稳
        mode = self.mode_by_ch.get(ch.nid, 'direct')
        # —— 记录一次全局成败（避免按观察者重复计数） ——
        if mode == 'direct':
            self._dir_t += 1
            if ok: self._dir_s += 1
        else:
            self._rel_t += 1
            if ok: self._rel_s += 1
        for _w in watchers:
            if ok and timely:
                # 成功：降低可疑，增加可靠性记忆
                ch.suspicion = max(0.0, ch.suspicion - 0.05)
                ch.observed_success += 0.4
                if mode == 'direct':
                    ch.dir_val = 0.88 * ch.dir_val + 0.12 * 1.0;
                    ch.dir_cnt += 1
                else:
                    ch.rly_val = 0.88 * ch.rly_val + 0.12 * 1.0;
                    ch.rly_cnt += 1
            else:
                # 失败：有限度地增加可疑，并轻微衰减可靠性
                if random.random() < 0.70:
                    ch.observed_fail += 0.6
                    ch.suspicion = min(1.0, ch.suspicion + 0.20)
                    ch.consecutive_strikes += 1
                ch.dir_val *= 0.93;
                ch.rly_val *= 0.93

    def finalize_trust_blacklist(self):
        sim=self.sim
        for n in sim.alive_nodes():
            n.trust_s = n.trust_s * self.forget + n.observed_success
            n.trust_f = n.trust_f * self.forget + n.observed_fail + 0.20 * n.suspicion
            # 更稳健：需要“至少一次击穿”配合低信任，或击穿累计超阈值；亦或高可疑且出现过击穿
            if (n.trust() < self.trust_blacklist and n.consecutive_strikes >= 1) or \
                    (n.consecutive_strikes >= self.strike_threshold) or \
                    (n.suspicion >= 0.85 and n.consecutive_strikes >= 1):
                n.blacklisted = True
        # —— 全局自适应：基于 direct/relay 的近端PDR，微调两跳阈值与冗余强度 ——
        alive_now = sim.alive_nodes()
        tot = self._rel_t + self._dir_t
        # 每轮至少积累一定样本再调整（避免抖动）
        if tot >= max(10, int(0.2 * len(alive_now) + 1)):
            rel_pdr = (self._rel_s / max(1, self._rel_t))
            dir_pdr = (self._dir_s / max(1, self._dir_t))
            if rel_pdr > dir_pdr + 0.03:
                # relay 明显更稳：适度放宽 alpha，并略增冗余强度
                self.alpha = min(self.alpha_max, self.alpha + self.alpha_step)
                self.rt_ratio = min(self.rt_max, self.rt_ratio * 1.05)
            elif dir_pdr > rel_pdr + 0.03:
                # direct 更稳：收紧 alpha，并略降冗余强度
                self.alpha = max(self.alpha_min, self.alpha - self.alpha_step)
                self.rt_ratio = max(self.rt_min, self.rt_ratio * 0.95)
            # 指数遗忘，保持灵敏但不震荡
            self._rel_s *= 0.5; self._rel_t *= 0.5
            self._dir_s *= 0.5; self._dir_t *= 0.5
