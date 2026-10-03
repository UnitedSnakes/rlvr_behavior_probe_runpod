#!/usr/bin/env python3
"""Independent fact-check recompute for camera_ready/RESULTS.md and BRIDGE_REPORT.md.

Written from scratch from PREREG_RUNS.md, PREREG_BRIDGE.md and the paper's
definitions. Uses only numpy/json/os/math. Does not import anything from
camera_ready/analysis or analyses/.

Modes:
  box      ROOT PTC OUT     all box-side recomputation (raw banks, evals, ledgers, step logs)
  laptop20 ROOT20 BANKDIR OUT   F1 counts over all 20 seed-42 snapshots (data only on the laptop)
"""
import json, math, os, sys, glob
import numpy as np

EVAL_STEPS = [934, 1681, 2428, 3736]
PCT = {934: '025', 1681: '045', 2428: '065', 3736: '100'}
BINS = ['0', '(0,.25]', '(.25,.5]', '(.5,.75]', '(.75,1)']
METRICS = ['R', 'T', 'C']
# Student t quantiles (hard-coded; numpy only)
T975 = {1: 12.706204736174705, 2: 4.302652729911275, 3: 3.182446305284263, 4: 2.776445105197799}
T95 = {1: 6.313751514675043, 2: 2.919985580355516, 3: 2.353363434801823, 4: 2.131846786326649}
DELTA = 3.0


def bin_of(p):
    if p == 0:
        return 0
    if p <= 0.25:
        return 1
    if p <= 0.5:
        return 2
    if p <= 0.75:
        return 3
    if p < 1:
        return 4
    return 5


# ----------------------------------------------------------------------------- loaders
def load_bank(d):
    """A/B 16-response pi0 banks. Returns dict half -> dict of (256,16) arrays."""
    out = {h: {k: np.full((256, 16), np.nan) for k in ('R', 'T', 'C', 'L')} for h in 'AB'}
    seen = set()
    for f in sorted(glob.glob(os.path.join(d, 'rollouts_shard*of2.jsonl'))):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                i = r['dataset_index']
                if i >= 256:
                    continue
                assert i not in seen
                seen.add(i)
                for h in 'AB':
                    ro = r['rollouts_' + h]
                    assert len(ro) == 16
                    for j, x in enumerate(ro):
                        R = float(x['canonical_reward']); T = float(bool(x['terminated'])); C = float(bool(x['correct']))
                        assert R == T * C, (f, i, h, j)
                        out[h]['R'][i, j] = R; out[h]['T'][i, j] = T; out[h]['C'][i, j] = C
                        out[h]['L'][i, j] = float(x['n_tokens'])
    assert seen == set(range(256)), len(seen)
    return out


def load_snapshot(path, seed=None):
    R = np.full((256, 16), np.nan); T = R.copy(); C = R.copy(); L = R.copy()
    seen = set()
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            i = r['dataset_index']
            assert 0 <= i < 256 and i not in seen
            seen.add(i)
            if seed is not None:
                assert r['question_seed'] == seed * 100000 + i + 75000, (path, i, r['question_seed'])
            ro = r['rollouts']
            assert len(ro) == 16, (path, i, len(ro))
            for j, x in enumerate(ro):
                rr = float(x['canonical_reward']); tt = float(bool(x['terminated'])); cc = float(bool(x['correct']))
                assert rr == tt * cc
                R[i, j] = rr; T[i, j] = tt; C[i, j] = cc; L[i, j] = float(x['completion_length'])
            assert r['n_correct'] == int(C[i].sum()) and r['n_terminated'] == int(T[i].sum()) and r['n_reward'] == int(R[i].sum())
    assert seen == set(range(256))
    return {'R': R, 'T': T, 'C': C, 'L': L}


def load_ledger(d, objective):
    files = sorted(glob.glob(os.path.join(d, '*.jsonl')))
    groups = {}  # (step, rank) -> dict
    rows = 0
    nonfinite = 0
    sums = {'tok_s': 0.0, 'tok_q': 0.0, 'tok_n': 0.0, 'act_s': 0.0, 'act_q': 0.0, 'act_n': 0.0}
    for f in files:
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                rows += 1
                for k, v in r.items():
                    if isinstance(v, float) and not math.isfinite(v):
                        nonfinite += 1
                key = (int(r['generation_global_step']), int(r['rank']))
                g = groups.setdefault(key, {'idx': set(), 'r': [], 'a': [], 'gs': set(), 'gstd': [], 'gsz': set()})
                g['idx'].add(int(r['dataset_index']))
                g['r'].append(float(r['canonical_reward']))
                g['a'].append(float(r['advantage']))
                g['gs'].add(r.get('group_successes'))
                g['gsz'].add(r.get('group_size'))
                sums['tok_s'] += r['token_ratio_sum']; sums['tok_q'] += r['token_ratio_sq_sum']; sums['tok_n'] += r['token_delta_count']
                if r.get('actual_is_ratio_sum') is not None:
                    sums['act_s'] += r['actual_is_ratio_sum']; sums['act_q'] += r['actual_is_ratio_sq_sum']; sums['act_n'] += r['actual_is_ratio_count']
    maxerr = 0.0
    bad = 0
    gl = []
    for (s, rk), g in groups.items():
        if len(g['idx']) != 1 or len(g['r']) != 16:
            bad += 1
            continue
        r = np.array(g['r']); a = np.array(g['a'])
        if objective == 'grpo':
            rec = (r - r.mean()) / (r.std(ddof=1) + 1e-4)
        else:
            K = r.sum()
            rec = np.zeros(16) if K == 0 else (r - K / 16.0) / (K / 16.0)
        maxerr = max(maxerr, float(np.max(np.abs(a - rec))))
        gl.append((s, rk, next(iter(g['idx'])), float(np.abs(a).sum()), bool(np.any(a != 0)), int(r.sum())))
    gl.sort()
    steps = sorted(set(x[0] for x in gl))
    per_step_rank = {}
    for s, rk, *_ in gl:
        per_step_rank.setdefault(s, set()).add(rk)
    integ = {
        'rank_files': len(files), 'rows': rows, 'groups': len(groups), 'bad_groups': bad,
        'steps': len(steps), 'step_min': steps[0], 'step_max': steps[-1],
        'steps_contiguous': steps == list(range(steps[0], steps[-1] + 1)),
        'every_step_two_ranks': all(len(v) == 2 for v in per_step_rank.values()),
        'max_advantage_error': maxerr, 'nonfinite_numeric_fields': nonfinite,
        'ess_token_ratio': sums['tok_s'] ** 2 / (sums['tok_q'] * sums['tok_n']),
        'ess_actual_ratio': (sums['act_s'] ** 2 / (sums['act_q'] * sums['act_n'])) if sums['act_n'] else None,
    }
    return gl, integ


def load_steplog(path):
    gn = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if 'grad_norm' in r and r.get('grad_norm') is not None and 'step' in r:
                gn[int(r['step'])] = float(r['grad_norm'])
    return gn


# ----------------------------------------------------------------------------- statistics
def bin_membership(bank):
    """Frozen bins per direction from mean canonical reward of that half."""
    return {h: np.array([bin_of(bank[h]['R'][i].mean()) for i in range(256)]) for h in 'AB'}


def bin_weights(binsd):
    """(5,256) row-normalized indicator for each direction."""
    W = {}
    for h in 'AB':
        M = np.zeros((5, 256))
        for b in range(5):
            m = binsd[h] == b
            M[b, m] = 1.0 / m.sum()
        W[h] = M
    return W


def rates(snap):
    return {X: snap[X].mean(axis=1) for X in METRICS}


def dq(rM, rG, W):
    """d_b and q_b (pp) for one metric given per-question rates."""
    diff = rM - rG
    d = 0.5 * (W['A'] @ diff + W['B'] @ diff)
    panel = diff.mean()
    return 100 * d, 100 * (d - panel)


def tstats(vals, one_sided=False):
    v = np.asarray(vals, float)
    n = len(v)
    m = v.mean()
    out = {'n': n, 'mean': m, 'values': v.tolist()}
    if n >= 3:
        se = v.std(ddof=1) / math.sqrt(n)
        out['ci'] = [m - T975[n - 1] * se, m + T975[n - 1] * se]
        out['upper95'] = m + T95[n - 1] * se
    return out


def wording(st):
    if 'ci' not in st:
        return 'unresolved at this number of runs'
    lo, hi = st['ci']
    if lo > 0:
        return 'additional gain under MaxRL'
    if hi < 0:
        return 'lower under MaxRL'
    if lo > -DELTA and hi < DELTA:
        return 'no additional gain larger than δ pp detected'
    return 'unresolved at this number of runs'


def cond_boot(snapM, snapG, W, metrics, ndraw=3000, seed=20261002, chunk=250):
    """F1 conditional sampling intervals: resample each question's 16 responses per arm."""
    rng = np.random.default_rng(seed)
    res = {X: {'d': [], 'q': []} for X in metrics}
    ii = np.arange(256)[None, :, None]
    done = 0
    while done < ndraw:
        nb = min(chunk, ndraw - done)
        iM = rng.integers(0, 16, size=(nb, 256, 16))
        iG = rng.integers(0, 16, size=(nb, 256, 16))
        for X in metrics:
            rM = snapM[X][ii, iM].mean(axis=2)
            rG = snapG[X][ii, iG].mean(axis=2)
            diff = rM - rG  # (nb,256)
            d = 0.5 * (diff @ W['A'].T + diff @ W['B'].T)
            q = d - diff.mean(axis=1, keepdims=True)
            res[X]['d'].append(100 * d); res[X]['q'].append(100 * q)
        done += nb
    out = {}
    for X in metrics:
        out[X] = {}
        for s in ('d', 'q'):
            a = np.concatenate(res[X][s])
            out[X][s] = np.percentile(a, [2.5, 97.5], axis=0).T.tolist()  # per bin [lo,hi]
    return out


def exposure_steps(gl):
    s = {}
    for step, rk, idx, m, live, K in gl:
        if idx < 256:
            s.setdefault(idx, []).append(step)
    return s


def mass(gl, binsd, t, clip=None):
    """Symmetric S_t(b) per bin and Tot_t."""
    per_q = np.zeros(256)
    tot = 0.0
    for step, rk, idx, m, live, K in gl:
        if step < t:
            w = m if clip is None else m * clip[step + 1]
            tot += w
            if idx < 256:
                per_q[idx] += w
    S = np.zeros(5)
    for h in 'AB':
        for b in range(5):
            mm = binsd[h] == b
            S[b] += 0.5 * per_q[mm].sum() / mm.sum()
    return S, tot


def ols_adjusted(y, E, Z):
    n = len(y)
    cols = [np.ones(n), E.astype(float)]
    dropped = 0
    for j in range(Z.shape[1]):
        x = Z[:, j]
        sd = x.std(ddof=1)
        if not (sd > 0):
            dropped += 1
            continue
        cols.append((x - x.mean()) / sd)
    X = np.column_stack(cols)
    beta, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    return beta[0] + beta[1], beta[0], int(rank) == X.shape[1], dropped


def pre_exposure(Ct, bank, binsd, s_q, ptc, cut, b):
    """Frozen adjustment model for one cutoff and bin. Returns E, U, U-E (pp), counts."""
    res = {}
    for h, o in (('A', 'B'), ('B', 'A')):
        m = binsd[h] == b
        idx = np.where(m)[0]
        y = 100 * (Ct[idx] - bank[o]['C'][idx].mean(axis=1))
        E = np.array([s_q[i] < cut for i in idx])
        Z = np.column_stack([bank[o]['R'][idx].mean(axis=1), bank[o]['L'][idx].mean(axis=1), ptc[idx]])
        nE, nU = int(E.sum()), int((~E).sum())
        if nE < 2 or nU < 2:
            res[h] = {'skipped': True, 'nE': nE, 'nU': nU}
            continue
        aE, aU, fullrank, dropped = ols_adjusted(y, E, Z)
        res[h] = {'E': aE, 'U': aU, 'nE': nE, 'nU': nU, 'fullrank': fullrank, 'dropped': dropped}
    ok = [h for h in 'AB' if not res[h].get('skipped')]
    out = {'dirs': res}
    if ok:
        out['E'] = float(np.mean([res[h]['E'] for h in ok]))
        out['U'] = float(np.mean([res[h]['U'] for h in ok]))
        out['UmE'] = out['U'] - out['E']
        out['ndirs'] = len(ok)
    return out


def clip_coef(gn):
    steps = sorted(gn)
    n = np.array([gn[s] for s in steps])
    c = np.minimum(1.0, 1.0 / (n + 1e-6))
    q = np.percentile(c, [0, 5, 25, 50, 75, 95, 100])
    return {'steps_logged': len(steps), 'share_c_lt_1': float(np.mean(c < 1)), 'share_gn_gt_1': float(np.mean(n > 1.0)),
            'n_gn_eq_1': int(np.sum(n == 1.0)), 'mean': float(c.mean()), 'quantiles': q.tolist(),
            'first_step': steps[0], 'last_step': steps[-1]}, dict(zip(steps, c))


def batch_structure(gl, binsd):
    live_by_step = {}
    panel_by_step = {}
    for step, rk, idx, m, live, K in gl:
        live_by_step[step] = live_by_step.get(step, 0) + int(live)
        if idx < 256:
            panel_by_step.setdefault(step, []).append(idx)
    steps = sorted(live_by_step)
    nl = np.array([live_by_step[s] for s in steps])
    allsh = [float(np.mean(nl == k)) for k in (0, 1, 2)]
    perbin = []
    perbin_qweighted = []
    for b in range(5):
        acc = np.zeros(3); accq = np.zeros(3)
        for h in 'AB':
            st = [s for s in steps if any(binsd[h][q] == b for q in panel_by_step.get(s, []))]
            v = np.array([live_by_step[s] for s in st])
            acc += 0.5 * np.array([np.mean(v == k) for k in (0, 1, 2)])
            stq = [s for s in steps for q in panel_by_step.get(s, []) if binsd[h][q] == b]
            vq = np.array([live_by_step[s] for s in stq])
            accq += 0.5 * np.array([np.mean(vq == k) for k in (0, 1, 2)])
        perbin.append(acc.tolist()); perbin_qweighted.append(accq.tolist())
    return {'all': allsh, 'perbin_steps': perbin, 'perbin_question_weighted': perbin_qweighted}


def panel_rates(snap):
    return {X: 100 * snap[X].mean() for X in METRICS}


# ----------------------------------------------------------------------------- box mode
def mtime(p):
    try:
        return os.stat(p).st_mtime
    except OSError:
        return None


def run_box(root, ptc_path, out_path):
    out = {}
    with open(ptc_path) as fh:
        ptcd = json.load(fh)
    ptc = np.array([float(ptcd[str(i)]) for i in range(256)])

    a40 = load_bank(os.path.join(root, 'banks/a40_original'))
    boxb = load_bank(os.path.join(root, 'bridge_box/pi0_bank'))
    binsd = bin_membership(a40)
    W = bin_weights(binsd)
    out['bin_counts'] = {h: [int(np.sum(binsd[h] == b)) for b in range(6)] for h in 'AB'}
    # paper App B check: rates in opposite bank for empirical-zero bin
    out['zero_bin_opposite'] = {
        'A_bin_B_base_R': 100 * a40['B']['R'][binsd['A'] == 0].mean(), 'A_bin_B_base_C': 100 * a40['B']['C'][binsd['A'] == 0].mean(),
        'B_bin_A_base_R': 100 * a40['A']['R'][binsd['B'] == 0].mean(), 'B_bin_A_base_C': 100 * a40['A']['C'][binsd['B'] == 0].mean()}
    pi0_pooled = {X: 100 * np.concatenate([a40['A'][X], a40['B'][X]], axis=1).mean() for X in METRICS}
    out['pi0_panel_a40'] = pi0_pooled
    # F5 bin contents (opposite bank), directions averaged
    f5 = []
    for b in range(5):
        vals = []
        for h, o in (('A', 'B'), ('B', 'A')):
            m = binsd[h] == b
            Cq = a40[o]['C'][m].mean(axis=1); Tq = a40[o]['T'][m].mean(axis=1)
            vals.append([100 * Cq.mean(), 100 * Tq.mean(), 100 * np.mean(Cq >= 0.5)])
        f5.append(np.mean(vals, axis=0).tolist())
    out['F5'] = f5

    # ------------------------------------------------------------------ pairs
    pairs = {'seed43': 43, 'seed44': 44, 'seed45': 45, 'seed42_a40': 42}
    snaps = {}
    for pn, sd in pairs.items():
        for arm in ('grpo', 'maxrl'):
            for t in EVAL_STEPS:
                p = os.path.join(root, pn, arm, 'eval', 'pi_' + PCT[t], 'snapshot_raw.jsonl')
                snaps[(pn, arm, t)] = load_snapshot(p, seed=sd)
    out['pairs'] = {}
    ledgers = {}
    for pn, sd in pairs.items():
        P = {'panel': {}, 'dq': {}, 'mass': {}, 'integrity': {}, 'clip': {}, 'clip_mass': {}, 'pre_exposure': {}}
        for arm in ('grpo', 'maxrl'):
            P['panel'][arm] = {t: panel_rates(snaps[(pn, arm, t)]) for t in EVAL_STEPS}
            gl, integ = load_ledger(os.path.join(root, pn, arm, 'ledger'), arm)
            ledgers[(pn, arm)] = gl
            sq = exposure_steps(gl)
            integ['panel_sampled'] = len(sq)
            integ['panel_sampled_once'] = all(len(v) == 1 for v in sq.values())
            integ['training_questions'] = len(set(x[2] for x in gl))
            P['integrity'][arm] = integ
            sl = os.path.join(root, pn, arm, 'step_log.jsonl')
            if os.path.exists(sl):
                gn = load_steplog(sl)
                cs, cmap = clip_coef(gn)
                P['clip'][arm] = cs
                P['clip_map_' + arm] = cmap
                # file-time facts
                lj = os.path.join(root, pn, arm, 'camera_ready_launch.json')
                with open(lj) as fh:
                    L = json.load(fh)
                P['integrity'][arm]['gpus'] = L['env'].get('CUDA_VISIBLE_DEVICES')
                P['integrity'][arm]['training_commit'] = L.get('training_commit')
                tl = os.path.join(root, pn, arm, arm + '_train_attempt1.log')
                t0, t1 = mtime(lj), mtime(tl)
                P['integrity'][arm]['wall_h_launch_to_log'] = (t1 - t0) / 3600 if t0 and t1 else None
                # max allocated memory
                mx = 0.0
                with open(sl) as fh:
                    for line in fh:
                        r = json.loads(line)
                        mx = max(mx, float(r.get('cr_max_memory_allocated_gib') or 0))
                P['integrity'][arm]['max_mem_alloc_gib'] = mx
                evt = {}
                for t in EVAL_STEPS:
                    dd = os.path.join(root, pn, arm, 'eval', 'pi_' + PCT[t])
                    a, b_ = mtime(os.path.join(dd, 'camera_ready_eval_provenance.json')), mtime(os.path.join(dd, 'snapshot_raw.jsonl'))
                    lg = mtime(os.path.join(root, pn, arm, 'eval', 'pi_' + PCT[t] + '.log'))
                    evt[t] = {'raw_minus_prov_min': (b_ - a) / 60 if a and b_ else None, 'log_minus_prov_min': (lg - a) / 60 if a and lg else None}
                P['integrity'][arm]['eval_minutes'] = evt
            P['sched_' + arm] = sq
        # same schedule across arms
        sqG, sqM = P.pop('sched_grpo'), P.pop('sched_maxrl')
        P['same_schedule_panel'] = all(sqG[i] == sqM[i] for i in sqG) and set(sqG) == set(sqM)
        s_q = np.array([sqG[i][0] if i in sqG else np.inf for i in range(256)])
        # d, q
        for t in EVAL_STEPS:
            rM, rG = rates(snaps[(pn, 'maxrl', t)]), rates(snaps[(pn, 'grpo', t)])
            P['dq'][t] = {}
            for X in METRICS:
                d, q = dq(rM[X], rG[X], W)
                P['dq'][t][X] = {'d': d.tolist(), 'q': q.tolist()}
        # within-pair conditional intervals at the endpoint (and all steps for discovery)
        steps_ci = EVAL_STEPS if pn == 'seed42_a40' else [3736]
        P['cond_ci'] = {}
        for t in steps_ci:
            P['cond_ci'][t] = cond_boot(snaps[(pn, 'maxrl', t)], snaps[(pn, 'grpo', t)], W, METRICS)
        # mass
        glG, glM = ledgers[(pn, 'grpo')], ledgers[(pn, 'maxrl')]
        for t in EVAL_STEPS:
            SG, TG = mass(glG, binsd, t); SM, TM = mass(glM, binsd, t)
            P['mass'][t] = {'SG': SG.tolist(), 'SM': SM.tolist(), 'TotG': TG, 'TotM': TM,
                            'ratio': (SM / SG).tolist(), 'share_ratio': ((SM / TM) / (SG / TG)).tolist()}
            if 'clip_map_grpo' in P:
                cG, cM = P['clip_map_grpo'], P['clip_map_maxrl']
                SGc, _ = mass(glG, binsd, t, clip=cG); SMc, _ = mass(glM, binsd, t, clip=cM)
                P['clip_mass'][t] = (SMc / SGc).tolist()
        P.pop('clip_map_grpo', None); P.pop('clip_map_maxrl', None)
        # pre-exposure (GRPO arm), all bins, A40 baselines
        for cut in (934, 1681, 2428):
            Ct = snaps[(pn, 'grpo', cut)]['C'].mean(axis=1)
            P['pre_exposure'][cut] = {}
            for b in range(5):
                P['pre_exposure'][cut][b] = pre_exposure(Ct, a40, binsd, s_q, ptc, cut, b)
        # bin-level Delta C for discovery endpoint (paper Table 2)
        if pn == 'seed42_a40':
            dC = {}
            for arm in ('grpo', 'maxrl'):
                Cq = snaps[(pn, arm, 3736)]['C'].mean(axis=1)
                v = []
                for b in range(5):
                    acc = 0
                    for h, o in (('A', 'B'), ('B', 'A')):
                        m = binsd[h] == b
                        acc += 0.5 * 100 * (Cq[m] - a40[o]['C'][m].mean(axis=1)).mean()
                    v.append(acc)
                dC[arm] = v
            P['endpoint_deltaC'] = dC
            F4 = {arm: batch_structure(ledgers[(pn, arm)], binsd) for arm in ('grpo', 'maxrl')}
            P['F4'] = F4
        if pn != 'seed42_a40':
            F4 = {arm: batch_structure(ledgers[(pn, arm)], binsd) for arm in ('grpo', 'maxrl')}
            P['F4'] = F4
        out['pairs'][pn] = P
        # free ledgers after use
        ledgers.pop((pn, 'grpo')); ledgers.pop((pn, 'maxrl'))
        print('done', pn, flush=True)

    # seed-42 MaxRL clipping from the A40 trainer_state (every 10th step)
    ts = '/root/autodl-tmp/a40/maxrl_trainer_meta/trainer_state.json'
    if os.path.exists(ts):
        with open(ts) as fh:
            lh = json.load(fh)['log_history']
        gn = {int(r['step']): float(r['grad_norm']) for r in lh if 'grad_norm' in r}
        out['seed42_maxrl_clip'] = clip_coef(gn)[0]

    # ------------------------------------------------------------------ across-pair stats
    new = ['seed43', 'seed44', 'seed45']
    prim = {}
    for X in ('C', 'R'):
        for b in (0, 1):
            for s in ('d', 'q'):
                vals = [out['pairs'][p]['dq'][3736][X][s][b] for p in new]
                st = tstats(vals)
                st['wording'] = wording(st)
                disc = out['pairs']['seed42_a40']['dq'][3736][X][s][b]
                st['discovery'] = disc
                st['discovery_pos'] = 'inside' if min(vals) <= disc <= max(vals) else ('below' if disc < min(vals) else 'above')
                mix = tstats(vals + [disc])
                st['mixed'] = mix
                prim[f'{s}_{X}_{b}'] = st
    out['primary'] = prim
    sec = {}
    for X in METRICS:
        for t in EVAL_STEPS:
            for s in ('d', 'q'):
                for b in range(5):
                    sec[f'{X}_{t}_{s}_{b}'] = tstats([out['pairs'][p]['dq'][t][X][s][b] for p in new])
    out['secondary'] = sec
    out['reweighting'] = {
        'endpoint': all(out['pairs'][p]['mass'][3736]['ratio'][b] > 1 for p in new for b in (0, 1)),
        'all_steps': all(out['pairs'][p]['mass'][t]['ratio'][b] > 1 for p in new for t in EVAL_STEPS for b in (0, 1))}
    out['replicated'] = all(out['pairs'][p]['pre_exposure'][c][1]['U'] > 0 for p in new for c in (934, 1681, 2428))
    out['random_schedule'] = {c: tstats([out['pairs'][p]['pre_exposure'][c][1]['UmE'] for p in new]) for c in (934, 1681, 2428)}

    # discovery sensitivity row: seed-42 checkpoints re-evaluated on the box
    sens = {}
    for t in EVAL_STEPS:
        bM = load_snapshot(os.path.join(root, 'bridge_box/maxrl_seed42/pi_' + PCT[t], 'snapshot_raw.jsonl'), seed=42)
        bG = load_snapshot(os.path.join(root, 'bridge_box/grpo_seed42/pi_' + PCT[t], 'snapshot_raw.jsonl'), seed=42)
        snaps[('box42', 'maxrl', t)] = bM; snaps[('box42', 'grpo', t)] = bG
        d, q = dq(bM['C'].mean(axis=1), bG['C'].mean(axis=1), W)
        sens[t] = {'d': d.tolist(), 'q': q.tolist()}
    out['discovery_box_sensitivity'] = sens

    # ------------------------------------------------------------------ bridge
    ck = [('pi0', None)] + [(a, t) for a in ('grpo', 'maxrl') for t in EVAL_STEPS]
    delta = {}  # name -> X -> per-question diff (256,)
    extra = {}
    for a, t in ck:
        name = 'pi0' if a == 'pi0' else f'{a}_{t}'
        if a == 'pi0':
            A = {X: np.concatenate([a40['A'][X], a40['B'][X]], axis=1) for X in METRICS + ['L']}
            B = {X: np.concatenate([boxb['A'][X], boxb['B'][X]], axis=1) for X in METRICS + ['L']}
        else:
            A = snaps[('seed42_a40', a, t)]; B = snaps[('box42', a, t)]
        delta[name] = {X: B[X].mean(axis=1) - A[X].mean(axis=1) for X in METRICS}
        perbin = {}
        for X in METRICS:
            dd = delta[name][X]
            perbin[X] = (100 * 0.5 * (W['A'] @ dd + W['B'] @ dd)).tolist()
        extra[name] = {'len_a40': A['L'].mean(), 'len_box': B['L'].mean(),
                       'cap_a40': 100 * (1 - A['T']).mean(), 'cap_box': 100 * (1 - B['T']).mean(), 'perbin': perbin}
    br = {'cells': {}}
    for name in delta:
        br['cells'][name] = {}
        for X in METRICS:
            v = delta[name][X]
            D = 100 * v.mean(); SE = 100 * v.std(ddof=1) / 16.0
            z = 0.0 if (SE == 0 and D == 0) else D / SE
            br['cells'][name][X] = {'D': D, 'SE': SE, 'z': z}
        br['cells'][name]['maxz'] = max(abs(br['cells'][name][X]['z']) for X in METRICS)
        br['cells'][name].update(extra[name])
    # per-bank pi0
    br['pi0_bank'] = {h: {X: 100 * (boxb[h][X].mean(axis=1) - a40[h][X].mean(axis=1)).mean() for X in METRICS} for h in 'AB'}
    names = list(delta)
    D = {X: np.array([delta[n][X] for n in names]) for X in METRICS}  # (9,256)
    steps_both = EVAL_STEPS
    iM = [names.index(f'maxrl_{t}') for t in steps_both]; iG = [names.index(f'grpo_{t}') for t in steps_both]
    pooled = {X: 100 * D[X].mean() for X in METRICS}
    inter = {X: 100 * (D[X][iM].mean(axis=1) - D[X][iG].mean(axis=1)).mean() for X in METRICS}
    rng = np.random.default_rng(20261001)
    idx = rng.integers(0, 256, size=(5000, 256))
    Wb = np.zeros((5000, 256))
    for k in range(5000):
        Wb[k] = np.bincount(idx[k], minlength=256) / 256.0
    bp, bi = {}, {}
    for X in METRICS:
        per_ck = Wb @ D[X].T  # (5000,9)
        bp[X] = (100 * np.percentile(per_ck.mean(axis=1), [2.5, 97.5])).tolist()
        bi[X] = (100 * np.percentile((per_ck[:, iM] - per_ck[:, iG]).mean(axis=1), [2.5, 97.5])).tolist()
    br['pooled'] = {X: {'D': pooled[X], 'ci': bp[X]} for X in METRICS}
    br['interaction'] = {X: {'I': inter[X], 'ci': bi[X]} for X in METRICS}
    ga = all(-1.5 < bp[X][0] and bp[X][1] < 1.5 for X in METRICS)
    allz = [abs(br['cells'][n][X]['z']) for n in names for X in METRICS]
    nck25 = sum(1 for n in names if br['cells'][n]['maxz'] > 2.5)
    gb = max(allz) <= 3.5 and nck25 <= 1
    gc = all(abs(inter[X]) <= 1.5 and -3.0 < bi[X][0] and bi[X][1] < 3.0 for X in ('C', 'R'))
    gd = sum(1 for n in names if n != 'pi0') >= 4 and any(n.startswith('grpo') for n in names) and any(n.startswith('maxrl') for n in names)
    gross = any(abs(br['cells']['pi0'][X]['D']) > 5 or abs(pooled[X]) > 5 for X in METRICS)
    br['gates'] = {'a': ga, 'b': gb, 'b_maxz': max(allz), 'b_n_ck_gt_2_5': nck25, 'c': gc, 'd': gd, 'gross_fail': gross}
    br['verdict'] = 'PASS' if (ga and gb and gc and gd and not gross and len(names) == 9) else ('GROSS FAIL' if gross else 'NOT PASSED')
    out['bridge'] = br

    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (float(o) if isinstance(o, np.floating) else (int(o) if isinstance(o, np.integer) else str(o))))
    print('WROTE', out_path, flush=True)


# ----------------------------------------------------------------------------- laptop20 mode
def run_laptop20(root20, bankdir, out_path):
    a40 = load_bank(bankdir)
    binsd = bin_membership(a40)
    W = bin_weights(binsd)
    res = {}
    for k in range(1, 21):
        pct = '%03d' % (5 * k)
        G = load_snapshot(os.path.join(root20, 'grpo/eval/pi_' + pct, 'snapshot_raw.jsonl'), seed=42)
        M = load_snapshot(os.path.join(root20, 'maxrl/eval/pi_' + pct, 'snapshot_raw.jsonl'), seed=42)
        ci = cond_boot(M, G, W, ['C'])
        d, q = dq(M['C'].mean(axis=1), G['C'].mean(axis=1), W)
        res[pct] = {'ci': ci['C'], 'd': d.tolist(), 'q': q.tolist()}
        print('snap', pct, flush=True)
    counts = {}
    for s in ('d', 'q'):
        counts[s] = []
        for b in range(5):
            above = sum(1 for p in res if res[p]['ci'][s][b][0] > 0)
            below = sum(1 for p in res if res[p]['ci'][s][b][1] < 0)
            margin = min(min(abs(res[p]['ci'][s][b][0]), abs(res[p]['ci'][s][b][1])) for p in res)
            counts[s].append({'above': above, 'below': below, 'closest_endpoint_to_0': margin})
    with open(out_path, 'w') as fh:
        json.dump({'per_snapshot': res, 'counts': counts}, fh, indent=1)
    print('WROTE', out_path)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'box':
        run_box(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == 'laptop20':
        run_laptop20(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode != 'compare':
        raise SystemExit('unknown mode')


# ============================================================================= compare mode
# Parses the tables of RESULTS.md / BRIDGE_REPORT.md and compares every number with
# the recomputed values (fc_box.json from `box`, fc_laptop20.json from `laptop20`).
import re

NUM = re.compile(r'[+\-−]?\d[\d,]*(?:\.\d+)?(?:e[+\-]?\d+)?')


def nums(s):
    return [float(x.replace('−', '-').replace(',', '')) for x in NUM.findall(s)]


def dec(s):
    """number of decimals of each number token in s"""
    out = []
    for x in NUM.findall(s):
        x = x.split('e')[0]
        out.append(len(x.split('.')[1]) if '.' in x else 0)
    return out


def table_after(lines, marker, start=0):
    i = next(k for k in range(start, len(lines)) if marker in lines[k])
    j = i if lines[i].startswith('|') else next(k for k in range(i + 1, len(lines)) if lines[k].startswith('|'))
    rows = []
    k = j + 2  # skip header + separator
    while k < len(lines) and lines[k].startswith('|'):
        rows.append([c.strip() for c in lines[k].strip().strip('|').split('|')])
        k += 1
    return rows, k


class Checker:
    def __init__(self):
        self.items = []

    def num(self, where, theirs, mine, kind='pp', ndec=2, tol=None):
        """kind: 'pp' exact pp (tol 0.01), 'rel' (1e-3 rel or display rounding), 'boot' (tol given)."""
        if mine is None:
            self.items.append((where, theirs, None, 'NOT RECOMPUTED'))
            return
        disp = round(mine, ndec)
        if kind == 'pp':
            ok = abs(theirs - mine) <= 0.01 + 1e-9
        elif kind == 'rel':
            ok = abs(theirs - disp) <= 0.5 * 10 ** (-ndec) + 1e-12 or abs(theirs - mine) <= 1e-3 * abs(mine)
        else:
            ok = abs(theirs - mine) <= tol
        self.items.append((where, theirs, mine, 'MATCH' if ok else 'MISMATCH'))

    def eq(self, where, theirs, mine):
        self.items.append((where, theirs, mine, 'MATCH' if theirs == mine else 'MISMATCH'))


def boot_tol(lo, hi, ndraw=3000):
    """MC tolerance for comparing two independent percentile endpoints (3.5 SE of the difference)."""
    sigma = max((hi - lo) / 3.92, 1e-9)
    se = 0.0488 * sigma * math.sqrt(3000.0 / ndraw)
    return max(0.15, 3.5 * math.sqrt(2) * se)


def run_compare(fc_box, fc20, results_md, bridge_md, out_json):
    o = json.load(open(fc_box)); f20 = json.load(open(fc20))
    L = open(results_md).read().split('\n')
    BL = open(bridge_md).read().split('\n')
    ck = Checker()
    P = o['pairs']
    new = ['seed43', 'seed44', 'seed45']
    pname = {'seed43': 'seed43', 'seed44': 'seed44', 'seed45': 'seed45', 'seed 42 (A40)': 'seed42_a40'}
    binkey = {'0': 0, '(0,.25]': 1}

    # ---- validity table
    rows, _ = table_after(L, '| Seed | GPUs')
    for r in rows:
        p = 'seed' + r[0]
        I = P[p]['integrity']
        g, m = [x.strip() for x in r[1].split('/')]
        ck.eq(f'validity {p} GPUs grpo', g, I['grpo']['gpus']); ck.eq(f'validity {p} GPUs maxrl', m, I['maxrl']['gpus'])
        th = nums(r[2])
        ck.num(f'validity {p} training h grpo (file mtimes)', th[0], I['grpo']['wall_h_launch_to_log'], 'boot', tol=0.02)
        ck.num(f'validity {p} training h maxrl (file mtimes)', th[1], I['maxrl']['wall_h_launch_to_log'], 'boot', tol=0.02)
        rgs = nums(r[3])
        for arm in ('grpo', 'maxrl'):
            ck.eq(f'validity {p} rows/groups/steps {arm}', rgs, [I[arm]['rows'], I[arm]['groups'], I[arm]['steps']])
        e = nums(r[4])
        ck.num(f'validity {p} max adv err grpo', e[0], I['grpo']['max_advantage_error'], 'rel', ndec=9)
        ck.num(f'validity {p} max adv err maxrl', e[1], I['maxrl']['max_advantage_error'], 'rel', ndec=9)
        s = nums(r[5])
        ck.num(f'validity {p} ESS/N grpo', s[0], I['grpo']['ess_actual_ratio'], 'rel', ndec=5)
        ck.num(f'validity {p} ESS/N maxrl', s[1], I['maxrl']['ess_actual_ratio'], 'rel', ndec=5)
        gn = nums(r[6])
        ck.eq(f'validity {p} grad norm logged', gn, [P[p]['clip']['grpo']['steps_logged'], P[p]['clip']['maxrl']['steps_logged']])

    # ---- prose panel rates at 3736 (line "Whole-panel rates at step 3736")
    line = next(x for x in L if x.startswith('Whole-panel rates at step 3736'))
    v = nums(line.split('C):')[1].replace('(A40)', ''))
    order = [('seed43', 'grpo'), ('seed43', 'maxrl'), ('seed44', 'grpo'), ('seed44', 'maxrl'), ('seed45', 'grpo'), ('seed45', 'maxrl'), ('seed42_a40', 'grpo'), ('seed42_a40', 'maxrl')]
    vv = [x for x in v if x not in (43, 44, 45, 42)]
    for k, (p, arm) in enumerate(order):
        for j, X in enumerate(METRICS):
            ck.num(f'prose panel 3736 {p} {arm} {X}', vv[3 * k + j], P[p]['panel'][arm]['3736'][X])

    # ---- primary table
    rows, _ = table_after(L, '| Cell | seed43 | seed44 | seed45 | Mean')
    for r in rows:
        s, X, b = [x.strip() for x in r[0].split(', ', 2)]
        b = binkey[b.replace('bin ', '')]
        key = f'{s}_{X}_{b}'
        pr = o['primary'][key]
        for j, p in enumerate(new):
            n_ = nums(r[1 + j])
            ck.num(f'primary {key} {p} value', n_[0], P[p]['dq']['3736'][X][s][b])
            ci = P[p]['cond_ci']['3736'][X][s][b]
            ck.num(f'primary {key} {p} within-pair lo', n_[1], ci[0], 'boot', tol=boot_tol(*ci))
            ck.num(f'primary {key} {p} within-pair hi', n_[2], ci[1], 'boot', tol=boot_tol(*ci))
        ck.num(f'primary {key} mean', nums(r[4])[0], pr['mean'])
        c = nums(r[5]); ck.num(f'primary {key} CI lo', c[0], pr['ci'][0]); ck.num(f'primary {key} CI hi', c[1], pr['ci'][1])
        ck.num(f'primary {key} one-sided upper', nums(r[6])[0], pr['upper95'])
        ck.eq(f'primary {key} wording', r[7], pr['wording'])
        ck.num(f'primary {key} discovery value', nums(r[8])[0], pr['discovery'])
        pos = 'inside' if 'inside' in r[8] else ('below' if 'below' in r[8] else 'above')
        ck.eq(f'primary {key} discovery position', pos, pr['discovery_pos'])

    # ---- sentences
    for x in L:
        m = re.match(r'- ([dq]), ([CR]), bin (0|\(0,\.25\]): mean (\S+) pp, 95 % CI \[(\S+), (\S+)\], one-sided 95 % upper bound (\S+) pp \(n = (\d)\) — \*\*(.+)\*\*', x)
        if m:
            key = f'{m.group(1)}_{m.group(2)}_{binkey[m.group(3)]}'
            pr = o['primary'][key]
            ck.num(f'sentence {key} mean', nums(m.group(4))[0], pr['mean'])
            ck.num(f'sentence {key} CI lo', nums(m.group(5))[0], pr['ci'][0]); ck.num(f'sentence {key} CI hi', nums(m.group(6))[0], pr['ci'][1])
            ck.num(f'sentence {key} upper', nums(m.group(7))[0], pr['upper95'])
            ck.eq(f'sentence {key} n', int(m.group(8)), pr['n'])
            ck.eq(f'sentence {key} wording', m.group(9), pr['wording'])

    # ---- mass table
    rows, _ = table_after(L, '| Pair | Step | 0 ratio; share ratio')
    for r in rows:
        p = pname[r[0]]; t = r[1]
        M = P[p]['mass'][t]
        for b in range(5):
            a = nums(r[2 + b])
            ck.num(f'mass {p} {t} bin{b} ratio', a[0], M['ratio'][b], 'rel')
            ck.num(f'mass {p} {t} bin{b} share ratio', a[1], M['share_ratio'][b], 'rel')

    # ---- reweighting flags
    line = next(x for x in L if x.startswith('Reweighting realized'))
    ck.eq('reweighting realized at 3736', 'at step 3736 in every new pair): **True**' in line, o['reweighting']['endpoint'])
    ck.eq('reweighting realized all steps', line.rstrip().endswith('True.'), o['reweighting']['all_steps'])

    # ---- pre-exposure table
    rows, _ = table_after(L, '| Pair | Cutoff | ΔC exposed')
    for r in rows:
        p = pname[r[0]]; c = r[1]
        pe = P[p]['pre_exposure'][c]['1']
        ck.num(f'pre-exposure {p} {c} E', nums(r[2])[0], pe['E'])
        ck.num(f'pre-exposure {p} {c} U', nums(r[3])[0], pe['U'])
        ck.num(f'pre-exposure {p} {c} U-E', nums(r[4])[0], pe['UmE'])
        ck.eq(f'pre-exposure {p} {c} counts', [int(x) for x in nums(r[5])],
              [pe['dirs']['A']['nE'], pe['dirs']['A']['nU'], pe['dirs']['B']['nE'], pe['dirs']['B']['nU']])
    line = next(x for x in L if x.startswith('Pre-exposure gain replicated'))
    ck.eq('pre-exposure replicated flag', '**True**' in line, o['replicated'])

    # ---- random schedule
    rows, _ = table_after(L, '| Cutoff | Per seed | Mean')
    for r in rows:
        c = r[0]; rs = o['random_schedule'][c]
        for j, x in enumerate(nums(r[1])):
            ck.num(f'random-schedule {c} seed{43 + j}', x, rs['values'][j])
        ck.num(f'random-schedule {c} mean', nums(r[2])[0], rs['mean'])
        ci = nums(r[3]); ck.num(f'random-schedule {c} CI lo', ci[0], rs['ci'][0]); ck.num(f'random-schedule {c} CI hi', ci[1], rs['ci'][1])

    # ---- mixed
    rows, _ = table_after(L, '| Cell | Values (box pairs')
    for r in rows:
        s, X, b = [x.strip() for x in r[0].split(', ', 2)]
        key = f'{s}_{X}_{binkey[b.replace("bin ", "")]}'
        mx = o['primary'][key]['mixed']
        for j, x in enumerate(nums(r[1])):
            ck.num(f'mixed {key} value{j}', x, mx['values'][j])
        ck.eq(f'mixed {key} n', int(nums(r[2])[0]), mx['n'])
        ck.num(f'mixed {key} mean', nums(r[3])[0], mx['mean'])
        ci = nums(r[4]); ck.num(f'mixed {key} CI lo', ci[0], mx['ci'][0]); ck.num(f'mixed {key} CI hi', ci[1], mx['ci'][1])

    # ---- secondary
    start = next(k for k in range(len(L)) if L[k].startswith('### Secondary: d_b and q_b'))
    for X in ('C', 'R', 'T'):
        rows, start = table_after(L, f'**{X}**', start)
        for r in rows:
            t, s = r[0], r[1]
            for b in range(5):
                a = nums(r[2 + b])
                st = o['secondary'][f'{X}_{t}_{s}_{b}']
                ck.num(f'secondary {X} {t} {s} bin{b} mean', a[0], st['mean'])
                ck.num(f'secondary {X} {t} {s} bin{b} CI lo', a[1], st['ci'][0]); ck.num(f'secondary {X} {t} {s} bin{b} CI hi', a[2], st['ci'][1])
                for j in range(3):
                    ck.num(f'secondary {X} {t} {s} bin{b} {new[j]}', a[3 + j], st['values'][j])

    # ---- panel table
    rows, _ = table_after(L, '| Pair | Arm | Step | R | T | C |')
    for r in rows:
        p = pname[r[0]]; arm = r[1].lower(); t = r[2]
        for j, X in enumerate(METRICS):
            ck.num(f'panel {p} {arm} {t} {X}', float(r[3 + j]), P[p]['panel'][arm][t][X])

    # ---- clipping
    rows, _ = table_after(L, '| Pair | Arm | Steps logged')
    for r in rows:
        p = r[0]; arm = r[1]; c = P[p]['clip'][arm]
        ck.eq(f'clip {p} {arm} steps logged', int(nums(r[2])[0]), c['steps_logged'])
        ck.num(f'clip {p} {arm} share clipped (c<1) %', nums(r[3])[0], 100 * c['share_c_lt_1'], 'rel', ndec=1)
        ck.num(f'clip {p} {arm} mean coef', nums(r[4])[0], c['mean'], 'rel', ndec=3)
        ck.num(f'clip {p} {arm} median', nums(r[5])[0], c['quantiles'][3], 'rel', ndec=3)
        ck.num(f'clip {p} {arm} p5', nums(r[6])[0], c['quantiles'][1], 'rel', ndec=3)
    rows, _ = table_after(L, 'Clip-weighted |A| mass ratio')
    for r in rows:
        p = r[0]; t = r[1]
        for b in range(5):
            ck.num(f'clip-weighted mass {p} {t} bin{b}', float(r[2 + b]), P[p]['clip_mass'][t][b], 'rel')

    # ---- discovery sensitivity row
    rows, _ = table_after(L, '| Step | Stat | Source |')
    for r in rows:
        t, s, src = r[0], r[1], r[2]
        mine = P['seed42_a40']['dq'][t]['C'][s] if src == 'A40' else o['discovery_box_sensitivity'][t][s]
        for b in range(5):
            ck.num(f'discovery sensitivity {t} {s} {src} bin{b}', nums(r[3 + b])[0], mine[b])

    # ---- F1/F2 tables (discovery, 4 steps)
    start = next(k for k in range(len(L)) if L[k].startswith('### F. Seed-42'))
    D = P['seed42_a40']
    for X in ('C', 'R', 'T'):
        rows, start = table_after(L, f'**F1/F2 — d_b and q_b for {X}', start)
        for r in rows:
            t, s = r[0], r[1]
            for b in range(5):
                a = nums(r[2 + b])
                ck.num(f'F1/F2 {X} {t} {s} bin{b} value', a[0], D['dq'][t][X][s][b])
                ci = D['cond_ci'][t][X][s][b]
                ck.num(f'F1/F2 {X} {t} {s} bin{b} lo', a[1], ci[0], 'boot', tol=boot_tol(*ci))
                ck.num(f'F1/F2 {X} {t} {s} bin{b} hi', a[2], ci[1], 'boot', tol=boot_tol(*ci))
    # F1 counts over 20 snapshots
    rows, _ = table_after(L, '**F1 across all 20 snapshots')
    for r in rows:
        s = r[0]
        for b in range(5):
            a = [int(x) for x in nums(r[1 + b])]
            cc = f20['counts'][s][b]
            ck.eq(f'F1 20-snapshot counts {s} bin{b} (above,below)', a, [cc['above'], cc['below']])
    # F3
    rows, _ = table_after(L, '**F3 — advantage mass')
    for r in rows:
        t = r[0]
        for b in range(5):
            a = nums(r[1 + b])
            ck.num(f'F3 {t} bin{b} ratio', a[0], D['mass'][t]['ratio'][b], 'rel')
            ck.num(f'F3 {t} bin{b} share ratio', a[1], D['mass'][t]['share_ratio'][b], 'rel')
    line = next(x for x in L if x.startswith('Total |A| over all training groups'))
    a = nums(line.split('MaxRL):')[1])
    for k, t in enumerate(EVAL_STEPS):
        ck.eq(f'F3 total |A| {t} (t, GRPO, MaxRL; rounded)', a[3 * k:3 * k + 3], [t, round(D['mass'][str(t)]['TotG']), round(D['mass'][str(t)]['TotM'])])
    # F4
    rows, _ = table_after(L, '**F4 — batch structure')
    for r in rows:
        arm = r[0].lower()
        F4 = D['F4'][arm]
        a = nums(r[1])
        for k in range(3):
            ck.num(f'F4 {arm} all steps share{k}', a[k], F4['all'][k], 'rel', ndec=3)
        for b in range(5):
            a = nums(r[2 + b])
            for k in range(3):
                ck.num(f'F4 {arm} bin{b} share{k} (step-level)', a[k], F4['perbin_steps'][b][k], 'rel', ndec=3)
    # F5
    rows, _ = table_after(L, '**F5 — bin contents')
    for b in range(5):
        a = nums(rows[0][b])
        for k, nm in enumerate(('mean C', 'mean T', 'share C>=.5')):
            ck.num(f'F5 bin{b} {nm} %', a[k], o['F5'][b][k], 'rel', ndec=1)
    # seed-42 clipping
    line = next(x for x in L if x.startswith('**Clipping (seed 42).**'))
    a = nums(line)
    c42 = o['seed42_maxrl_clip']
    # numbers: 42, 10, 373, 1, 78.0, 0.710, 0.719, 5, 0.315, 42, 42
    ck.eq('seed-42 MaxRL clip logged steps', int(a[2]), c42['steps_logged'])
    ck.num('seed-42 MaxRL share clipped, as labelled (grad norm > 1) %', a[4], 100 * c42['share_gn_gt_1'], 'rel', ndec=1)
    ck.num('seed-42 MaxRL share clipped, c<1 definition %', a[4], 100 * c42['share_c_lt_1'], 'rel', ndec=1)
    ck.num('seed-42 MaxRL mean coef', a[5], c42['mean'], 'rel', ndec=3)
    ck.num('seed-42 MaxRL median', a[6], c42['quantiles'][3], 'rel', ndec=3)
    ck.num('seed-42 MaxRL p5', a[8], c42['quantiles'][1], 'rel', ndec=3)

    # ---- bridge report
    B = o['bridge']
    bname = {'π0 (both banks, 32/question)': 'pi0'}
    for a_ in ('GRPO', 'MaxRL'):
        for t in EVAL_STEPS:
            bname[f'{a_} seed 42, step {t}'] = f'{a_.lower()}_{t}'
    rows, _ = table_after(BL, '| Checkpoint | ΔR (SE)')
    for r in rows:
        n_ = bname[r[0]]; c = B['cells'][n_]
        for j, X in enumerate(('R', 'T', 'C')):
            a = nums(r[1 + j])
            ck.num(f'bridge {n_} Δ{X}', a[0], c[X]['D']); ck.num(f'bridge {n_} SE{X}', a[1], c[X]['SE'])
        ck.num(f'bridge {n_} max|Δ/SE|', nums(r[4])[0], c['maxz'])
    rows, _ = table_after(BL, '| Metric | Pooled Δ')
    for r in rows:
        X = r[0]; a = nums(r[1]); ci = nums(r[2])
        ck.num(f'bridge pooled Δ{X}', a[0], B['pooled'][X]['D'])
        ck.num(f'bridge pooled Δ{X} CI lo', ci[0], B['pooled'][X]['ci'][0], 'boot', tol=0.15)
        ck.num(f'bridge pooled Δ{X} CI hi', ci[1], B['pooled'][X]['ci'][1], 'boot', tol=0.15)
    rows, _ = table_after(BL, '| Metric | Interaction')
    for r in rows:
        X = r[0]; a = nums(r[1]); ci = nums(r[2])
        ck.num(f'bridge interaction {X}', a[0], B['interaction'][X]['I'])
        ck.num(f'bridge interaction {X} CI lo', ci[0], B['interaction'][X]['ci'][0], 'boot', tol=0.15)
        ck.num(f'bridge interaction {X} CI hi', ci[1], B['interaction'][X]['ci'][1], 'boot', tol=0.15)
    rows, _ = table_after(BL, '| Checkpoint | Mean length')
    for r in rows:
        n_ = bname[r[0]]; c = B['cells'][n_]
        a = nums(r[1]); b_ = nums(r[2])
        ck.num(f'bridge {n_} length A40', a[0], c['len_a40'], 'rel', ndec=1); ck.num(f'bridge {n_} length box', a[1], c['len_box'], 'rel', ndec=1)
        ck.num(f'bridge {n_} cap-hit A40 %', b_[0], c['cap_a40']); ck.num(f'bridge {n_} cap-hit box %', b_[1], c['cap_box'])
    rows, _ = table_after(BL, 'Per-bin ΔC (pp)')
    for r in rows:
        n_ = bname[r[0]]; c = B['cells'][n_]
        for b in range(5):
            ck.num(f'bridge per-bin ΔC {n_} bin{b}', nums(r[1 + b])[0], c['perbin']['C'][b])
    line = next(x for x in BL if 'Per-bank π0 Δ' in x)
    a = nums(line.split('Per-bank π0 Δ (pp):')[1])
    for k, (h, X) in enumerate([(h, X) for h in 'AB' for X in ('R', 'T', 'C')]):
        ck.num(f'bridge pi0 bank {h} Δ{X}', a[k], B['pi0_bank'][h][X])
    gl = next(x for x in BL if x.startswith('Gates:'))
    g = B['gates']
    ck.eq('bridge gate a', '(a) pooled Δ intervals inside ±1.5 pp: True' in gl, g['a'])
    ck.eq('bridge gate b', '(b) per-cell |Δ/SE| rule: True' in gl, g['b'])
    ck.eq('bridge gate c', '(c) arm interaction for C and R: True' in gl, g['c'])
    ck.eq('bridge gate d', '(d) ≥ 4 RL checkpoints from two arms: True' in gl, g['d'])
    ck.eq('bridge GROSS FAIL', 'GROSS FAIL: True' in gl, g['gross_fail'])
    ck.eq('bridge verdict', 'PASS' if any('## Verdict: **PASS**' in x for x in BL) else 'other', B['verdict'])

    out = {'n_items': len(ck.items), 'n_match': sum(1 for x in ck.items if x[3] == 'MATCH'),
           'mismatches': [x for x in ck.items if x[3] != 'MATCH'], 'items': ck.items}
    with open(out_json, 'w') as fh:
        json.dump(out, fh, indent=0, default=str)
    print('items', out['n_items'], 'match', out['n_match'], 'non-match', len(out['mismatches']))
    for x in out['mismatches']:
        print('  ', x)


if __name__ == '__main__' and sys.argv[1] == 'compare':
    run_compare(*sys.argv[2:7])
