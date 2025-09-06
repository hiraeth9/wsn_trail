# -*- coding: utf-8 -*-
import math, random, os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from collections import deque


SEED = 42
AREA_W, AREA_H = 100.0, 100.0
BS_POS = (50.0, 150.0)
COMM_RANGE = 30.0
CH_NEIGHBOR_RANGE = 60.0
INIT_ENERGY = 0.5

E_ELEC = 50e-9;
E_FS = 10e-12;
E_MP = 0.0013e-12
D0 = math.sqrt(E_FS / E_MP);
E_DA = 5e-9
DATA_PACKET_BITS = 4000;
CTRL_PACKET_BITS = 200

# —— 控制面能量记账开关（可关掉做 A/B 对比）——
ENABLE_HANDSHAKE_ENERGY = True
SIM_ROUNDS = 500
BASE_P_CH = 0.15


CH_COOLDOWN = int(1.0 / BASE_P_CH)

P_MAL_MEMBER_DROP = 0.25
P_MAL_CH_DROP = 0.60
P_MAL_CH_DELAY = 0.30


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def e_tx(bits: int, d: float) -> float:
    return E_ELEC * bits + (E_FS * bits * (d ** 2) if d < D0 else E_MP * bits * (d ** 4))


def e_rx(bits: int) -> float:
    return E_ELEC * bits


def clamp(x, lo, hi): return max(lo, min(hi, x))


@dataclass
class Node:
    nid: int;
    x: float;
    y: float;
    node_type: str
    energy: float = INIT_ENERGY
    trust_s: float = 0.0;
    trust_f: float = 0.0
    suspicion: float = 0.0
    consecutive_strikes: int = 0
    blacklisted: bool = False
    last_ch_round: int = -9999
    alive: bool = True
    is_ch: bool = False;
    cluster_id: Optional[int] = None
    observed_success: float = 0.0;
    observed_fail: float = 0.0
    queue_level: int = 0;
    last_queue_level: int = 0
    dir_cnt: int = 0;
    dir_val: float = 0.0
    rly_cnt: int = 0;
    rly_val: float = 0.0

    def pos(self): return (self.x, self.y)

    def trust(self): return (self.trust_s + 1.0) / (self.trust_s + self.trust_f + 2.0)

    def reset_round_flags(self):
        self.is_ch = False;
        self.cluster_id = None
        self.observed_success = 0.0;
        self.observed_fail = 0.0
        self.queue_level = 0;
        self.suspicion = max(0.0, self.suspicion * 0.9)


@dataclass
class Cluster:
    ch_id: int;
    members: List[int] = field(default_factory=list)


class AlgorithmBase:
    def __init__(self, sim: 'Simulation'): self.sim = sim

    @property
    def name(self) -> str: return "BaseAlgo"

    @property
    def trust_warn(self) -> float: raise NotImplementedError

    @property
    def trust_blacklist(self) -> float: raise NotImplementedError

    @property
    def forget(self) -> float: return 0.98

    @property
    def strike_threshold(self) -> int: return 3

    def suspicion_blacklist(self) -> Optional[float]: return None

    def select_cluster_heads(self): raise NotImplementedError

    def allow_member_redundancy(self, member: Node, ch: Node) -> bool: return False

    def choose_ch_relay(self, ch: Node, ch_nodes: List[Node]):
        return None, {}

    def apply_watchdog(self, ch: Node, ok: bool, timely: bool, ch_nodes: List[Node]): pass

    def finalize_trust_blacklist(self): raise NotImplementedError


class Simulation:
    def __init__(self, algo_ctor, n_nodes: int = 100, n_malicious: int = 30, seed: int = SEED):
        self.n_nodes = n_nodes;
        self.n_malicious = n_malicious
        self.bs = BS_POS;
        self.round = 0;
        self.seed = seed
        self.nodes: List[Node] = [];
        self.clusters: Dict[int, Cluster] = {}
        self.history = [];
        self.rand = random.Random(seed)
        self._init_nodes()
        self.total_drop = 0;
        self.total_delivered = 0;
        self.total_timely_delivered = 0
        self.malicious_delay_times = 0;
        self.malicious_drop_packets = 0
        self.member_energy_total = 0.0;
        self.member_energy_effective = 0.0
        self.total_energy_used = 0.0;
        self.total_control_bits = 0;
        self.total_hop_count = 0
        self.sum_cluster_size = 0;
        self.count_cluster_rounds = 0;
        self.sum_ch_count = 0
        self.FND = None;
        self.HND = None;
        self.LND = None
        self.algo: AlgorithmBase = algo_ctor(self)
        # >>> NEW: 端到端尝试计数（用于 total_p / pdr / drop_p）
        self.total_generated = 0
        # >>> ADDED: 误拉黑统计
        self.false_blacklist_nodes: set[int] = set()
        self.false_blacklist_events: int = 0
        # 用于每轮快照
        self._prev_blacklist_normals: set[int] = set()
        # === Functional Lifetime（功能寿命）配置与状态 ===
        self.func_pdr_threshold = 0.85  # 阈值：窗口内平均PDR必须≥0.85
        self.func_window = 100  # 滑动窗口长度（按轮数）
        self.func_warmup = 200  # 前 200 轮不判定功能寿命
        self._pdr_win = deque(maxlen=self.func_window)
        self.functional_lifetime = None  # 首次跌破阈值时的轮次

    def _init_nodes(self):
        # 用本仿真的 seed 固定拓扑（同一 seed 下跨不同比例可公平对比）
        rng = np.random.default_rng(self.seed)
        xs = rng.random(self.n_nodes) * AREA_W
        ys = rng.random(self.n_nodes) * AREA_H
        coords = np.column_stack([xs, ys])

        # 恶意集合“递增嵌套”：固定一个随机全排列，取前 k 个
        order = list(range(self.n_nodes))
        self.rand.shuffle(order)  # self.rand 已由 seed 播种
        mal = set(order[:self.n_malicious])

        for i in range(self.n_nodes):
            t = "malicious" if i in mal else "normal"
            self.nodes.append(Node(i, float(coords[i, 0]), float(coords[i, 1]), t))

    def alive_nodes(self):
        return [n for n in self.nodes if n.alive]

    def elect_cluster_heads(self):
        self.clusters = {}
        for n in self.nodes: n.reset_round_flags()
        self.algo.select_cluster_heads()
        if len(self.clusters) == 0:
            alive = [n for n in self.alive_nodes() if not n.blacklisted] or self.alive_nodes()
            if alive:
                ch = max(alive, key=lambda x: x.energy)
                ch.is_ch = True;
                ch.last_ch_round = self.round;
                self.clusters[ch.nid] = Cluster(ch.nid)

    # === NEW === 广播阶段（步骤 1/2）：CH 广播，成员接收
    def advertise_phase(self):
        if not ENABLE_HANDSHAKE_ENERGY:
            return
        alive = self.alive_nodes()
        chs = [n for n in alive if n.is_ch]
        if not chs:
            return
        for ch in chs:
            if not ch.alive:
                continue
            # 只对“潜在成员”（非CH、在通信半径内）做一次覆盖性广播
            receivers = [m for m in alive
                         if (not m.is_ch) and m.alive and dist(m.pos(), ch.pos()) <= COMM_RANGE]
            if not receivers:
                continue
            # 以最远成员距离近似一次广播所需发射功率
            dmax = max(dist(ch.pos(), r.pos()) for r in receivers)
            e_adv = e_tx(CTRL_PACKET_BITS, dmax)
            if ch.energy >= e_adv:
                # 簇头发广播
                ch.energy -= e_adv
                self.total_energy_used += e_adv
                self.total_control_bits += CTRL_PACKET_BITS  # 记一次控制面比特
                # 成员接收广播（逐个扣接收能量）
                rx_cost = e_rx(CTRL_PACKET_BITS)
                for r in receivers:
                    if not r.alive:
                        continue
                    if r.energy >= rx_cost:
                        r.energy -= rx_cost
                        self.total_energy_used += rx_cost
                    else:
                        r.alive = False
            else:
                ch.alive = False

    def assign_members(self):
        alive = self.alive_nodes();
        chs = [n for n in alive if n.is_ch]
        if not chs: return
        for n in alive:
            if n.is_ch: continue
            cands = []
            for ch in chs:
                if ch.trust() < self.algo.trust_blacklist: continue
                d = dist(n.pos(), ch.pos())
                if d <= COMM_RANGE: cands.append((d, ch))
            if not cands: n.cluster_id = None; continue
            cands.sort(key=lambda x: x[0]);
            # 新版：3/4 维持原逻辑；新增 5/6 确认/时隙
            d0, chosen = cands[0]
            n.cluster_id = chosen.nid

            # --- 3/4: 成员发送 JOIN，CH 接收 ---
            e_join = e_tx(CTRL_PACKET_BITS, d0)
            if n.energy >= e_join:
                n.energy -= e_join
                self.total_energy_used += e_join
                self.total_control_bits += CTRL_PACKET_BITS
                rx_cost = e_rx(CTRL_PACKET_BITS)
                if chosen.energy >= rx_cost:
                    chosen.energy -= rx_cost
                    self.total_energy_used += rx_cost
                else:
                    chosen.alive = False
                    continue
            else:
                n.alive = False
                continue

            # --- 5/6: CH 下发确认/时隙，成员接收 ---
            if ENABLE_HANDSHAKE_ENERGY:
                e_ack = e_tx(CTRL_PACKET_BITS, d0)
                if chosen.alive and chosen.energy >= e_ack:
                    chosen.energy -= e_ack
                    self.total_energy_used += e_ack
                    self.total_control_bits += CTRL_PACKET_BITS
                    rx_cost = e_rx(CTRL_PACKET_BITS)
                    if n.alive and n.energy >= rx_cost:
                        n.energy -= rx_cost
                        self.total_energy_used += rx_cost
                        # 只有确认成功才把成员加入簇
                        self.clusters[chosen.nid].members.append(n.nid)
                    else:
                        n.alive = False
                        continue
                else:
                    chosen.alive = False
                    continue
            else:
                # 若关闭控制面记账，维持原先行为：直接加入簇
                self.clusters[chosen.nid].members.append(n.nid)

    def transmit_round(self):
        alive=self.alive_nodes(); chs=[n for n in alive if n.is_ch]
        if not chs: return
        data_events={}; expected_by_ch={ch.nid:set() for ch in chs}
        for cl in self.clusters.values():
            ch=self.nodes[cl.ch_id]
            if not ch.alive: continue
            for mid in cl.members:
                m=self.nodes[mid]
                if not m.alive: continue
                # >>> NEW: 每个成员本轮1条业务 → 端到端一次尝试
                self.total_generated += 1
                allow=self.algo.allow_member_redundancy(m, ch)
                rec=self._member_send(m, ch, allow, chs)
                if mid not in data_events: data_events[mid]={'tx_paths':[], 'delivered':False,'timely':False}
                for rid in rec:
                    e_cost=e_tx(DATA_PACKET_BITS, dist(m.pos(), self.nodes[rid].pos()))
                    data_events[mid]['tx_paths'].append((e_cost, rid))
                    expected_by_ch[rid].add(mid)
        for cl in self.clusters.values():
            ch=self.nodes[cl.ch_id]
            if not ch.alive: continue
            expected=set(cl.members); got=expected_by_ch[ch.nid]; missing=expected-got
            for _ in missing:
                ch.observed_fail+=0.1; self.total_drop+=1
            for _ in got:
                ch.observed_success+=0.3
        delivered_by_ch={}
        for ch in chs:
            if not ch.alive: continue
            n_recv=ch.queue_level
            if n_recv>0:
                e_aggr=E_DA*DATA_PACKET_BITS*n_recv
                if ch.energy>=e_aggr: ch.energy-=e_aggr; self.total_energy_used+=e_aggr
                else: ch.alive=False; continue
            do_drop=False; do_delay=False
            if ch.node_type=="malicious":
                r=random.random()
                if r<P_MAL_CH_DROP: do_drop=True
                elif r<P_MAL_CH_DROP+P_MAL_CH_DELAY: do_delay=True
            ok=False; timely=False; hops=0
            if do_drop:
                self.malicious_drop_packets+=n_recv; self.total_drop+=n_recv
                ch.observed_fail+=1.0; ch.suspicion=min(1.0, ch.suspicion+0.3); ch.consecutive_strikes+=1
            else:
                relay,_meta=self.algo.choose_ch_relay(ch, chs)
                if relay is not None and relay.alive:
                    d1=dist(ch.pos(), relay.pos()); e1=e_tx(DATA_PACKET_BITS, d1)
                    if ch.energy>=e1 and relay.energy>=e_rx(DATA_PACKET_BITS):
                        ch.energy-=e1; self.total_energy_used+=e1
                        relay.energy-=e_rx(DATA_PACKET_BITS); self.total_energy_used+=e_rx(DATA_PACKET_BITS)
                        relay.queue_level += n_recv
                        d2=dist(relay.pos(), self.bs); e2=e_tx(DATA_PACKET_BITS, d2)
                        if relay.energy>=e2:
                            relay.energy-=e2; self.total_energy_used+=e2
                            ok=True; timely=(not do_delay); hops=2
                            if do_delay: self.malicious_delay_times+=1; ch.observed_fail+=0.5
                            else: ch.observed_success+=1.0; ch.consecutive_strikes=0
                        else:
                            relay.alive=False; self.total_drop+=n_recv
                    else:
                        ch.alive=False; self.total_drop+=n_recv
                else:
                    d_bs=dist(ch.pos(), self.bs); e_ch=e_tx(DATA_PACKET_BITS, d_bs)
                    if ch.energy>=e_ch:
                        ch.energy-=e_ch; self.total_energy_used+=e_ch
                        ok=True; timely=(not do_delay); hops=1
                        if do_delay: self.malicious_delay_times+=1; ch.observed_fail+=0.5
                        else: ch.observed_success+=1.0; ch.consecutive_strikes=0
                    else:
                        ch.alive=False; self.total_drop+=n_recv
            delivered_by_ch[ch.nid]=(ok,timely,hops)
            ch.last_queue_level=ch.queue_level
            self.algo.apply_watchdog(ch, ok, timely, chs)
        for mid,info in data_events.items():
            delivered=[]; timely_flag=False; hop_used=0
            for e_cost, ch_id in info['tx_paths']:
                ok,t,h = delivered_by_ch.get(ch_id,(False,False,0))
                if ok:
                    delivered.append(e_cost)
                    timely_flag = timely_flag or t
                    hop_used = max(hop_used, h)
            if delivered:
                info['delivered']=True; info['timely']=timely_flag
                self.total_delivered+=1
                if hop_used > 0:
                    self.total_hop_count += hop_used
                if timely_flag: self.total_timely_delivered+=1
                self.member_energy_effective += min(delivered)
        self.algo.finalize_trust_blacklist()

    def _member_send(self, m: Node, ch: Node, allow_redundancy: bool, chs: List[Node]):
        rec = []
        if m.node_type == "malicious" and random.random() < P_MAL_MEMBER_DROP:
            self.total_drop += 1
            return rec

        d = dist(m.pos(), ch.pos())
        e = e_tx(DATA_PACKET_BITS, d)

        if m.energy >= e:
            m.energy -= e
            self.total_energy_used += e
            # >>> NEW: 成员侧真实能耗（主路径）
            self.member_energy_total += e
            if ch.energy >= e_rx(DATA_PACKET_BITS):
                ch.energy -= e_rx(DATA_PACKET_BITS)
                self.total_energy_used += e_rx(DATA_PACKET_BITS)
                rec.append(ch.nid)
                ch.queue_level += 1
            else:
                ch.alive = False
        else:
            m.alive = False
            self.total_drop += 1
            return rec

        if allow_redundancy and chs:
            alts = []
            for other in chs:
                if other.nid == ch.nid or (not other.alive) or other.trust() < self.algo.trust_blacklist:
                    continue
                dd = dist(m.pos(), other.pos())
                if dd <= COMM_RANGE:
                    alts.append((dd, other))
            if alts:
                alts.sort(key=lambda x: x[0])
                dd, alt = alts[0]
                ee = e_tx(DATA_PACKET_BITS, dd)
                if m.energy >= ee:
                    m.energy -= ee
                    self.total_energy_used += ee
                    # >>> NEW: 成员侧真实能耗（冗余路径）
                    self.member_energy_total += ee
                    if alt.energy >= e_rx(DATA_PACKET_BITS):
                        alt.energy -= e_rx(DATA_PACKET_BITS)
                        self.total_energy_used += e_rx(DATA_PACKET_BITS)
                        rec.append(alt.nid)
                        alt.queue_level += 1
                    else:
                        alt.alive = False

        # >>> REMOVED: 不再使用 e * len(rec) 近似
        # self.member_energy_total += e * len(rec)
        return rec

    def update_lifetime(self):
        dead = sum(1 for n in self.nodes if not n.alive)
        if self.FND is None and dead >= 1: self.FND = self.round
        if self.HND is None and dead >= self.n_nodes // 2: self.HND = self.round
        if self.LND is None and dead >= self.n_nodes: self.LND = self.round

    def step(self):
        # 新增：打印轮次信息，只针对指定的5个算法
        target_algos = ["ACTAR-2024", "TRAIL (ours)", "DST-WOA-2024", "SHEER-2025", "TFSM-DPC-2024"]
        if self.algo.name in target_algos:
            print(f"Algorithm: {self.algo.name}, Round: {self.round}")
        self.round += 1
        prev_gen = self.total_generated
        prev_del = self.total_delivered

        # >>> ADDED: 轮次开始时快照——当前已被拉黑的 normal 节点
        prev_norm_bl = set(n.nid for n in self.nodes if (n.node_type == "normal" and n.blacklisted))


        if len(self.alive_nodes()) == 0:
            self.update_lifetime();
            return False
        self.elect_cluster_heads();
        # === NEW: 广播阶段（1/2）===
        self.advertise_phase();
        self.assign_members();
        self.transmit_round();
        self.update_lifetime()

        # === 计算本轮 PDR，并更新功能寿命 ===
        gen_this = self.total_generated - prev_gen
        del_this = self.total_delivered - prev_del
        if self.round > self.func_warmup and gen_this > 0:
            pdr_this = del_this / gen_this
            self._pdr_win.append(pdr_this)
            if (self.functional_lifetime is None) and (len(self._pdr_win) == self.func_window):
                if (sum(self._pdr_win) / self.func_window) < self.func_pdr_threshold:
                    self.functional_lifetime = self.round

        # >>> ADDED: 本轮 watchdog/黑名单后，计算“本轮新拉黑的正常节点”
        curr_norm_bl = set(n.nid for n in self.nodes if (n.node_type == "normal" and n.blacklisted))
        newly = curr_norm_bl - prev_norm_bl
        if newly:
            self.false_blacklist_events += len(newly)
            self.false_blacklist_nodes.update(newly)

        alive_cnt = sum(1 for n in self.nodes if n.alive)
        ch_cnt = sum(1 for n in self.nodes if n.alive and n.is_ch)
        timely = (self.total_timely_delivered / max(self.total_delivered, 1))
        self.sum_ch_count += ch_cnt
        if len(self.clusters) > 0:
            import numpy as _np
            # >>> CHANGED: 原来是 mean(len(members))，现在改为 len(members)+1（含CH）
            csize = _np.mean([len(cl.members) + 1 for cl in self.clusters.values()])
            self.sum_cluster_size += csize;
            self.count_cluster_rounds += 1
        self.history.append({'round': self.round, 'alive': alive_cnt, 'chs': ch_cnt,
                             'cum_delivered': self.total_delivered, 'cum_drop': self.total_drop,
                             'cum_malicious_delay': self.malicious_delay_times,
                             'cum_malicious_drop': self.malicious_drop_packets,
                             'cum_timely_rate': timely})
        return (alive_cnt > 0)

    def run(self, rounds: int = SIM_ROUNDS):
        initE = sum(n.energy for n in self.nodes)
        if rounds is None or rounds <= 0:
            while self.step():
                pass
        else:
            for _ in range(rounds):
                if not self.step(): break
        finalE = sum(n.energy for n in self.nodes)
        # >>> CHANGED: 避免二次累加。直接用能量余额法覆盖总能耗。
        self.total_energy_used = (initE - finalE)

        bl_mal = sum(1 for n in self.nodes if (n.node_type == "malicious" and n.blacklisted))
        # >>> CHANGED: 正常节点被拉黑（去重）
        bl_norm_nodes = len(self.false_blacklist_nodes)
        # >>> ADDED: 误拉黑事件累计（跨轮新增次数）
        bl_norm_events = int(self.false_blacklist_events)
        # >>> ADDED: 端到端口径
        e2e_total = self.total_generated
        e2e_deliv = self.total_delivered
        e2e_drop = max(e2e_total - e2e_deliv, 0)
        # >>> REPLACE in run() 指标计算片段
        energy_rate = self.member_energy_effective / max(self.member_energy_total, 1.0)

        # >>> CHANGED: 全部用 e2e_* 口径
        timely_rate = self.total_timely_delivered / max(e2e_deliv, 1)
        pdr = e2e_deliv / max(e2e_total, 1)

        avg_ch = self.sum_ch_count / max(len(self.history), 1)
        avg_cluster = self.sum_cluster_size / max(self.count_cluster_rounds, 1)
        avg_hops = self.total_hop_count / max(e2e_deliv, 1)  # <<< 用 e2e_deliv
        energy_per_del = self.total_energy_used / max(e2e_deliv, 1)  # <<< 用 e2e_deliv
        throughput = (e2e_deliv * DATA_PACKET_BITS) / max(len(self.history), 1)  # <<< 用 e2e_deliv

        return ({
                    'algo': self.algo.name,
                    'FND': self.FND if self.FND is not None else self.round,
                    'HND': self.HND if self.HND is not None else self.round,
                    'LND': self.LND if self.LND is not None else self.round,
                    'func_life_pdr85': int(self.functional_lifetime) if self.functional_lifetime is not None else int(
                        self.round),
                    # >>> CHANGED: 端到端口径
                    'drop_p': int(e2e_drop),
                    'total_p': int(e2e_total),
                    'pdr': float(pdr),
                    'drop_rate': float(1.0 - pdr),  # >>> NEW: 与CSV一致
                    'timely_transfer_rate': float(timely_rate),
                    'energy_rate': float(energy_rate),
                    'energy_per_delivered': float(energy_per_del),
                    'throughput_bits_per_round': float(throughput),
                    'avg_ch_per_round': float(avg_ch),
                    'avg_cluster_size': float(avg_cluster),  # 口径已在 step() 修正为含CH
                    'avg_hops_to_bs': float(avg_hops),
                    'malicious_delay': int(self.malicious_delay_times),
                    'malicious_drop': int(self.malicious_drop_packets),
                    'blacklisted_malicious': int(bl_mal),
                    # >>> CHANGED: 原列 blacklisted_normal 现在表示“去重节点数”
                    'blacklisted_normal': int(bl_norm_nodes),
                    # >>> ADDED: 新增一列，表示“事件数”
                    'false_blacklist_events': bl_norm_events,
                    'control_overhead_bits': int(self.total_control_bits),
                    'rounds_run': self.round
                }, pd.DataFrame(self.history))
