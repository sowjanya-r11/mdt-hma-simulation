"""
MDT-HMA simulation: Multi-Datacenter Threshold-based Health Monitoring and Alerting
Synthetic evaluation of the Khalifa University / OpManager scenario
(1,500 devices, 6 data centers, 2 of them from merged organizations).

Run:  python mdt_hma_simulation.py
Needs: numpy
"""
import json
import math
import time
from collections import defaultdict

import numpy as np

DAYS = 3
T = DAYS * 1440                      # one sample per minute
N_SEEDS = 5
TYPES = ['switch'] * 1000 + ['server'] * 400 + ['router'] * 60 + ['firewall'] * 40
N = len(TYPES)
DC_P = [.22, .22, .18, .14, .12, .12]  # DC-1..DC-6 (DC-4, DC-5 = merged orgs A, B)
MERGED = {3, 4}
METRICS = ['cpu', 'mem', 'disk', 'loss', 'lat']

BASE = {  # cpu mean, cpu sd, mem mean, mem sd
    'switch': (25, 5, 40, 3), 'server': (38, 7, 52, 4),
    'router': (30, 5, 45, 3), 'firewall': (35, 5, 45, 3)}

# benign one-sample transients and random poll failures (per sample, per device)
P_TRANS = {'cpu': 1e-5, 'mem': 5e-6, 'loss': 1e-5, 'lat': 1e-5}
P_POLLFAIL = 2e-5

# injected faults: (count per 3 days, min dur, max dur, actionable level range, subtle level range)
FAULTS = {
    'cpu': (60, 10, 60, (82, 100), (65, 79)),
    'mem': (40, 30, 180, (82, 100), (65, 79)),
    'disk': (25, 120, 720, (87, 100), (72, 84)),
    'loss': (50, 5, 45, (2.5, 9), (1.0, 1.9)),
    'lat': (40, 10, 90, (160, 420), (100, 145)),
}
N_DOWN, DOWN_MIN, DOWN_MAX = 30, 3, 120
P_ACTIONABLE = 0.75

CONFIGS = {  # name: (thresholds, confirm polls for metrics, confirm polls for DOWN)
    'A  Case-study thresholds, single poll': (dict(cpu=80, mem=80, disk=85, loss=2, lat=150), 1, 1),
    'B  Case-study thresholds, 3-poll confirm': (dict(cpu=80, mem=80, disk=85, loss=2, lat=150), 3, 2),
    'C  Tight thresholds, single poll': (dict(cpu=70, mem=70, disk=75, loss=1, lat=120), 1, 1),
    'D  Tight thresholds, 3-poll confirm': (dict(cpu=70, mem=70, disk=75, loss=1, lat=120), 3, 2),
}
STATIC_WARN = CONFIGS['A  Case-study thresholds, single poll'][0]


def ar1(rng, phi):
    eps = rng.standard_normal((T, N), dtype=np.float32)
    x = np.empty((T, N), dtype=np.float32)
    x[0] = eps[0]
    a = np.float32(math.sqrt(1 - phi ** 2))
    for t in range(1, T):
        x[t] = phi * x[t - 1] + a * eps[t]
    return x


def build_world(seed):
    rng = np.random.default_rng(seed)
    types = np.array(TYPES)
    dc = rng.choice(6, size=N, p=DC_P)
    merged = np.isin(dc, list(MERGED))
    mu = {}
    sd = {}
    mu['cpu'] = np.array([BASE[t][0] for t in types]) + rng.normal(0, 3, N)
    sd['cpu'] = np.array([BASE[t][1] for t in types]) * rng.uniform(.7, 1.3, N)
    mu['mem'] = np.array([BASE[t][2] for t in types]) + rng.normal(0, 3, N)
    sd['mem'] = np.array([BASE[t][3] for t in types]) * rng.uniform(.7, 1.3, N)
    mu['disk'] = np.clip(rng.normal(55, 12, N), 20, 70)
    sd['disk'] = np.full(N, 1.0)
    mu['lat'] = np.where(merged, rng.uniform(40, 95, N), rng.uniform(15, 60, N))
    data = {}
    for m, phi in [('cpu', .9), ('mem', .95), ('disk', .99)]:
        data[m] = (mu[m] + sd[m] * ar1(rng, phi)).astype(np.float32)
    data['loss'] = rng.exponential(0.08, (T, N)).astype(np.float32)
    data['lat'] = (mu['lat'] * (1 + 0.08 * ar1(rng, .8))).astype(np.float32)

    # benign transients
    for m, p in P_TRANS.items():
        mask = rng.random((T, N)) < p
        k = int(mask.sum())
        if m == 'cpu' or m == 'mem':
            data[m][mask] = rng.uniform(85, 100, k)
        elif m == 'loss':
            data[m][mask] = rng.uniform(3, 8, k)
        else:
            data[m][mask] = rng.uniform(160, 400, k)

    reach = ~(rng.random((T, N)) < P_POLLFAIL)

    # injected faults
    w = 1 + merged.astype(float)
    w /= w.sum()
    faults = []
    fid = {m: np.full((T, N), -1, dtype=np.int32) for m in METRICS + ['down']}
    for m, (cnt, dmin, dmax, hi, lo) in FAULTS.items():
        for _ in range(rng.poisson(cnt)):
            d = rng.choice(N, p=w)
            dur = int(rng.integers(dmin, dmax + 1))
            t0 = int(rng.integers(0, T - dur))
            act = rng.random() < P_ACTIONABLE
            level = rng.uniform(*(hi if act else lo))
            seg = slice(t0, t0 + dur)
            data[m][seg, d] = np.maximum(data[m][seg, d], level + rng.normal(0, .02 * level, dur))
            fid[m][seg, d] = len(faults)
            faults.append(dict(metric=m, d=int(d), t0=t0, t1=t0 + dur, act=bool(act)))
    for _ in range(rng.poisson(N_DOWN)):
        d = rng.choice(N, p=w)
        dur = int(rng.integers(DOWN_MIN, DOWN_MAX + 1))
        t0 = int(rng.integers(0, T - dur))
        reach[t0:t0 + dur, d] = False
        fid['down'][t0:t0 + dur, d] = len(faults)
        faults.append(dict(metric='down', d=int(d), t0=t0, t1=t0 + dur, act=True))
    return dict(types=types, dc=dc, data=data, reach=reach, faults=faults, fid=fid, rng=rng)


def detect(breach, confirm):
    """Return (sample index, device) for the first poll of each confirmed breach run."""
    c = np.zeros(breach.shape[1], dtype=np.int16)
    ts, ds = [], []
    for t in range(breach.shape[0]):
        c = np.where(breach[t], c + 1, 0)
        hit = np.flatnonzero(c == confirm)
        if hit.size:
            ts.extend([t] * hit.size)
            ds.extend(hit.tolist())
    return ts, ds


def run_config(world, thr, conf_m, conf_d, k=1):
    alerts = []  # (time_min, device, metric)
    for m in METRICS:
        breach = (world['data'][m][::k] > thr[m]) & world['reach'][::k]
        ts, ds = detect(breach, conf_m)
        alerts += [(t * k, d, m) for t, d in zip(ts, ds)]
    ts, ds = detect(~world['reach'][::k], conf_d)
    alerts += [(t * k, d, 'down') for t, d in zip(ts, ds)]

    faults = world['faults']
    tp, fp = 0, 0
    first = {}
    by_dc = defaultdict(lambda: [0, 0, 0])
    by_type = defaultdict(lambda: [0, 0, 0])
    for t, d, m in alerts:
        f = world['fid'][m][min(t, T - 1), d]
        dc_label = int(world['dc'][d])
        ty = world['types'][d]
        by_dc[dc_label][0] += 1
        by_type[ty][0] += 1
        if f >= 0:
            tp += 1
            by_dc[dc_label][1] += 1
            by_type[ty][1] += 1
            if f not in first or t < first[f]:
                first[f] = t
        else:
            fp += 1
            by_dc[dc_label][2] += 1
            by_type[ty][2] += 1
    n_f = len(faults)
    act = [i for i, f in enumerate(faults) if f['act']]
    det_all = len(first)
    det_act = sum(1 for i in act if i in first)
    delays = [first[i] - faults[i]['t0'] for i in first]
    delays_act = [first[i] - faults[i]['t0'] for i in act if i in first]
    per_metric = {}
    for m in METRICS + ['down']:
        ids = [i for i, f in enumerate(faults) if f['metric'] == m]
        per_metric[m] = (sum(1 for i in ids if i in first), len(ids))
    return dict(
        alerts=len(alerts), tp=tp, fp=fp,
        precision=tp / max(1, len(alerts)),
        recall_all=det_all / max(1, n_f), recall_act=det_act / max(1, len(act)),
        mttd=float(np.mean(delays)) if delays else float('nan'),
        mttd_act=float(np.mean(delays_act)) if delays_act else float('nan'),
        med_act=float(np.median(delays_act)) if delays_act else float('nan'),
        fp_per_day=fp / DAYS, alerts_per_day=len(alerts) / DAYS,
        n_faults=n_f, n_act=len(act), per_metric=per_metric,
        by_dc=dict(by_dc), by_type=dict(by_type))


def manual_sweep(world, interval_min, rng):
    """Staff check each device once per sweep at a fixed (staggered) time; only visible faults are caught."""
    off = rng.uniform(0, interval_min, N)
    faults = world['faults']
    det, delays, n_act = 0, [], 0
    for f in faults:
        if not f['act']:
            continue
        n_act += 1
        o = off[f['d']]
        kk = 0 if f['t0'] <= o else math.ceil((f['t0'] - o) / interval_min)
        check = o + kk * interval_min
        if check < f['t1'] and check < T:
            det += 1
            delays.append(check - f['t0'])
    return det / n_act, float(np.mean(delays)) if delays else float('nan')


# ---------- merger reconciliation ----------
def make_inventory(n, rng):
    org = np.where(rng.random(n) < .26, rng.integers(1, 3, n), 0)  # 0 native, 1 org A, 2 org B
    ips, names = [], []
    used = set()
    for i in range(n):
        while True:
            if org[i] == 0:
                ip = f"10.{int(rng.integers(1, 4))}.{int(rng.integers(0, 256))}.{int(rng.integers(1, 255))}"
                nm = f"dev-{i:05d}"
            else:
                ip = f"172.{16 + int(org[i])}.{int(rng.integers(0, 256))}.{int(rng.integers(1, 255))}"
                nm = f"legacy{'AB'[org[i] - 1]}-{i:05d}"
            if ip not in used:
                used.add(ip)
                break
        ips.append(ip)
        names.append(nm)
    ips, names = np.array(ips, dtype=object), np.array(names, dtype=object)
    inj = set()
    merged_idx = np.flatnonzero(org > 0)
    scale = n / 1500
    for _ in range(round(15 * scale)):      # duplicate IP across orgs
        a = int(rng.choice(merged_idx))
        b = int(rng.integers(0, n))
        if org[b] != org[a]:
            ips[a] = ips[b]
            inj.add((min(a, b), max(a, b)))
    a_idx, b_idx = np.flatnonzero(org == 1), np.flatnonzero(org == 2)
    for _ in range(round(10 * scale)):      # duplicate hostname between orgs A and B
        a, b = int(rng.choice(a_idx)), int(rng.choice(b_idx))
        names[a] = names[b]
        inj.add((min(a, b), max(a, b)))
    return org, ips, names, inj


def naive_pairs(org, ips, names):
    n = len(org)
    out = set()
    for i in range(n):
        oi, ii, ni = org[i], ips[i], names[i]
        for j in range(i + 1, n):
            if org[j] != oi and (ips[j] == ii or names[j] == ni):
                out.add((i, j))
    return out


def hash_pairs(org, ips, names):
    out = set()
    for key in (ips, names):
        groups = defaultdict(list)
        for i, v in enumerate(key):
            groups[v].append(i)
        for g in groups.values():
            if len(g) > 1:
                for x in range(len(g)):
                    for y in range(x + 1, len(g)):
                        if org[g[x]] != org[g[y]]:
                            out.add((g[x], g[y]))
    return out


def main():
    t_start = time.time()
    agg = defaultdict(lambda: defaultdict(list))
    first_world_detail = None
    sens = defaultdict(lambda: defaultdict(list))
    manual = defaultdict(lambda: defaultdict(list))
    ref_counts = []
    for seed in range(N_SEEDS):
        w = build_world(seed)
        ref_counts.append((len(w['faults']), sum(f['act'] for f in w['faults'])))
        for name, (thr, cm, cd) in CONFIGS.items():
            r = run_config(w, thr, cm, cd)
            for key in ('alerts_per_day', 'fp_per_day', 'precision', 'recall_all', 'recall_act', 'mttd_act', 'med_act'):
                agg[name][key].append(r[key])
            if name.startswith('B') and seed == 0:
                first_world_detail = r
        thrB, cmB, cdB = CONFIGS['B  Case-study thresholds, 3-poll confirm']
        for k in (1, 5, 15):
            r = run_config(w, thrB, cmB, cdB, k=k)
            for key in ('alerts_per_day', 'fp_per_day', 'recall_act', 'mttd_act'):
                sens[k][key].append(r[key])
        rng = np.random.default_rng(1000 + seed)
        for label, iv in (('manual_24h', 1440), ('manual_8h', 480)):
            rc, dl = manual_sweep(w, iv, rng)
            manual[label]['recall_act'].append(rc)
            manual[label]['mttd_act'].append(dl)
        print('seed', seed, 'done', round(time.time() - t_start, 1), 's')

    def ms(x):
        return float(np.nanmean(x)), float(np.nanstd(x))

    out = {'faults_per_run': ref_counts,
           'configs': {n: {k: ms(v) for k, v in d.items()} for n, d in agg.items()},
           'sensitivity': {k: {kk: ms(vv) for kk, vv in d.items()} for k, d in sens.items()},
           'manual': {k: {kk: ms(vv) for kk, vv in d.items()} for k, d in manual.items()},
           'detail_B_seed0': {
               'by_dc': {int(k): v for k, v in first_world_detail['by_dc'].items()},
               'by_type': first_world_detail['by_type'],
               'per_metric': first_world_detail['per_metric'],
               'recall_act': first_world_detail['recall_act'],
               'n_faults': first_world_detail['n_faults'], 'n_act': first_world_detail['n_act'],
               'alerts': first_world_detail['alerts'], 'fp': first_world_detail['fp']}}

    # devices per DC (seed 0)
    w0 = build_world(0)
    out['devices_per_dc'] = [int((w0['dc'] == i).sum()) for i in range(6)]
    out['type_by_dc'] = {t: [int(((w0['dc'] == i) & (w0['types'] == t)).sum()) for i in range(6)]
                         for t in ('switch', 'server', 'router', 'firewall')}

    # reconciliation benchmark
    rec = []
    for n in (750, 1500, 3000, 6000):
        rng = np.random.default_rng(7)
        org, ips, names, inj = make_inventory(n, rng)
        t0 = time.perf_counter(); a = naive_pairs(org, ips, names); t_naive = time.perf_counter() - t0
        t0 = time.perf_counter(); b = hash_pairs(org, ips, names); t_hash = time.perf_counter() - t0
        rec.append(dict(n=n, pairs=n * (n - 1) // 2, injected=len(inj), naive_found=len(a), hash_found=len(b),
                        same=(a == b), inj_recovered=len(inj & a), t_naive=t_naive, t_hash=t_hash))
    out['reconcile'] = rec
    out['runtime_s'] = time.time() - t_start
    with open('mdt_hma_results.json', 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
