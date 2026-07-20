"""
2026 FIFA World Cup probabilistic forecast
- Elo ratings computed from 49k international matches (1872-2026)
- Gradient boosting match-outcome model (W/D/L) for backtest calibration
- Poisson goal model for score simulation
- 10,000 Monte Carlo simulations of the real 2026 bracket (48 teams, R32 format)
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.linear_model import PoissonRegressor

rng = np.random.default_rng(42)

# ---------------- Load data ----------------
df = pd.read_csv('results.csv', parse_dates=['date'])
df = df.dropna(subset=['home_score', 'away_score']).copy()
df = df.sort_values('date').reset_index(drop=True)

def k_factor(tourn):
    t = str(tourn)
    if 'FIFA World Cup' in t and 'qualification' not in t: return 60
    if 'qualification' in t or 'UEFA Euro' in t or 'Copa' in t or 'Africa Cup' in t or 'AFC Asian Cup' in t or 'Gold Cup' in t: return 40
    if 'Friendly' in t: return 20
    return 30

# ---------------- Elo + features over full history ----------------
elo = defaultdict(lambda: 1500.0)
HOME_ADV = 80.0
recent_gf = defaultdict(list)  # rolling goals for
recent_ga = defaultdict(list)

rows = []
for r in df.itertuples():
    h, a = r.home_team, r.away_team
    eh, ea = elo[h], elo[a]
    home_bonus = 0.0 if r.neutral else HOME_ADV
    exp_h = 1.0 / (1.0 + 10 ** (-((eh + home_bonus) - ea) / 400.0))
    hs, as_ = int(r.home_score), int(r.away_score)
    res = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)

    formN = 10
    f_gf_h = np.mean(recent_gf[h][-formN:]) if recent_gf[h] else 1.3
    f_ga_h = np.mean(recent_ga[h][-formN:]) if recent_ga[h] else 1.3
    f_gf_a = np.mean(recent_gf[a][-formN:]) if recent_gf[a] else 1.3
    f_ga_a = np.mean(recent_ga[a][-formN:]) if recent_ga[a] else 1.3

    rows.append({
        'date': r.date, 'home': h, 'away': a, 'hs': hs, 'as': as_,
        'elo_h': eh, 'elo_a': ea, 'neutral': int(r.neutral),
        'elo_diff': (eh + home_bonus) - ea,
        'form_gf_h': f_gf_h, 'form_ga_h': f_ga_h,
        'form_gf_a': f_gf_a, 'form_ga_a': f_ga_a,
        'tournament': r.tournament,
        'outcome': 2 if hs > as_ else (1 if hs == as_ else 0),  # 2=home win,1=draw,0=away win
    })

    # Elo update with margin-of-victory multiplier
    K = k_factor(r.tournament)
    margin = abs(hs - as_)
    mov = 1.0 if margin <= 1 else (1.5 if margin == 2 else (11 + margin) / 8.0)
    delta = K * mov * (res - exp_h)
    elo[h] += delta
    elo[a] -= delta
    recent_gf[h].append(hs); recent_ga[h].append(as_)
    recent_gf[a].append(as_); recent_ga[a].append(hs)

md = pd.DataFrame(rows)
FEATS = ['elo_diff', 'neutral', 'form_gf_h', 'form_ga_h', 'form_gf_a', 'form_ga_a']

def brier_multiclass(y_true, proba):
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))

# ---------------- Backtests: 2018 and 2022 World Cups ----------------
print("=== BACKTESTS (train strictly on data before each tournament) ===")
for year, start, end in [(2018, '2018-06-14', '2018-07-15'), (2022, '2022-11-20', '2022-12-18')]:
    train = md[(md.date < start) & (md.date >= '1990-01-01')]
    test = md[(md.date >= start) & (md.date <= end) & (md.tournament == 'FIFA World Cup')]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=4, random_state=0)
    clf.fit(train[FEATS], train['outcome'])
    proba = clf.predict_proba(test[FEATS])
    ll = log_loss(test['outcome'], proba, labels=[0, 1, 2])
    br = brier_multiclass(test['outcome'].values, proba)
    # naive baseline: historical class frequencies
    base = train['outcome'].value_counts(normalize=True).sort_index().values
    base_p = np.tile(base, (len(test), 1))
    print(f"WC {year}: n={len(test)}  log-loss={ll:.4f} (baseline {log_loss(test['outcome'], base_p, labels=[0,1,2]):.4f})  "
          f"Brier={br:.4f} (baseline {brier_multiclass(test['outcome'].values, base_p):.4f})")

# ---------------- Final models trained on all data (1990+) ----------------
train = md[md.date >= '1990-01-01']
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=4, random_state=0)
clf.fit(train[FEATS], train['outcome'])

# Poisson goal models for score simulation (elo_diff scaled to ~unit range)
train = train.copy()
train['elo_diff_s'] = train['elo_diff'] / 400.0
GF = ['elo_diff_s', 'neutral', 'form_gf_h', 'form_ga_a']
GA = ['elo_diff_s', 'neutral', 'form_gf_a', 'form_ga_h']
ph = PoissonRegressor(alpha=1e-4, max_iter=2000).fit(train[GF], train['hs'])
pa = PoissonRegressor(alpha=1e-4, max_iter=2000).fit(train[GA], train['as'])
print('Poisson coefs (home goals):', dict(zip(GF, ph.coef_.round(3))))

# ---------------- 2026 setup ----------------
groups = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'B': ['Canada', 'Switzerland', 'Qatar', 'Bosnia and Herzegovina'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}
HOSTS = {'United States', 'Canada', 'Mexico'}
cur_elo = {t: elo[t] for g in groups.values() for t in g}
cur_gf = {t: (np.mean(recent_gf[t][-10:]) if recent_gf[t] else 1.3) for g in groups.values() for t in g}
cur_ga = {t: (np.mean(recent_ga[t][-10:]) if recent_ga[t] else 1.3) for g in groups.values() for t in g}

def match_lambdas(t1, t2):
    """Expected goals for t1 (as 'home' slot) and t2. Host nations get home advantage."""
    bonus = HOME_ADV if (t1 in HOSTS) else (-HOME_ADV if t2 in HOSTS else 0.0)
    ed = cur_elo[t1] - cur_elo[t2] + bonus
    neutral = 0 if (t1 in HOSTS or t2 in HOSTS) else 1
    xh = ph.predict(pd.DataFrame([{'elo_diff_s': ed/400.0, 'neutral': neutral, 'form_gf_h': cur_gf[t1], 'form_ga_a': cur_ga[t2]}]))[0]
    xa = pa.predict(pd.DataFrame([{'elo_diff_s': ed/400.0, 'neutral': neutral, 'form_gf_a': cur_gf[t2], 'form_ga_h': cur_ga[t1]}]))[0]
    return max(xh, 0.05), max(xa, 0.05)

LAM = {}
def sim_match(t1, t2):
    if (t1, t2) not in LAM:
        LAM[(t1, t2)] = match_lambdas(t1, t2)
    l1, l2 = LAM[(t1, t2)]
    return rng.poisson(l1), rng.poisson(l2)

def sim_knockout(t1, t2):
    g1, g2 = sim_match(t1, t2)
    if g1 != g2:
        return t1 if g1 > g2 else t2
    # extra time + penalties: weight slightly by elo
    p1 = 1.0 / (1.0 + 10 ** (-(cur_elo[t1] - cur_elo[t2]) / 400.0))
    p1 = 0.5 + 0.5 * (p1 - 0.5)  # shrink toward coin flip
    return t1 if rng.random() < p1 else t2

# Official R32 structure (winner-slot, opponent-slot). 3rd-place allocations approximated
# by drawing from each match's allowed group set.
R32 = [
    ('A2', 'B2'), ('C1', 'F2'), ('E1', '3:ABCDF'), ('F1', 'C2'),
    ('E2', 'I2'), ('I1', '3:CDFGH'), ('A1', '3:CEFHI'), ('L1', '3:EHIJK'),
    ('G1', '3:AEHIJ'), ('D1', '3:BEFIJ'), ('H1', 'J2'), ('K2', 'L2'),
    ('B1', '3:EFGIJ'), ('D2', 'G2'), ('J1', 'H2'), ('K1', '3:DEIJL'),
]

champions = defaultdict(int)
finalists = defaultdict(int)
semis = defaultdict(int)
N_SIM = 10000

for s in range(N_SIM):
    # group stage
    pos = {}        # 'A1' -> team
    thirds = {}     # group letter -> third-placed team
    for g, teams in groups.items():
        pts = {t: 0 for t in teams}
        gd = {t: 0 for t in teams}
        gf_ = {t: 0 for t in teams}
        for i in range(4):
            for j in range(i + 1, 4):
                t1, t2 = teams[i], teams[j]
                g1, g2 = sim_match(t1, t2)
                gd[t1] += g1 - g2; gd[t2] += g2 - g1
                gf_[t1] += g1; gf_[t2] += g2
                if g1 > g2: pts[t1] += 3
                elif g2 > g1: pts[t2] += 3
                else: pts[t1] += 1; pts[t2] += 1
        order = sorted(teams, key=lambda t: (pts[t], gd[t], gf_[t], rng.random()), reverse=True)
        pos[g + '1'], pos[g + '2'] = order[0], order[1]
        thirds[g] = (order[2], pts[order[2]], gd[order[2]], gf_[order[2]])
    # best 8 thirds advance
    ranked_thirds = sorted(thirds.items(), key=lambda kv: (kv[1][1], kv[1][2], kv[1][3], rng.random()), reverse=True)
    qualified_thirds = {g: v[0] for g, v in ranked_thirds[:8]}

    # fill R32, assigning qualified thirds to allowed slots
    available = dict(qualified_thirds)
    bracket = []
    third_slots = [(idx, spec[1][2:]) for idx, spec in enumerate(R32) if spec[1].startswith('3:')]
    assignment = {}
    for idx, allowed in third_slots:
        cands = [g for g in available if g in allowed]
        pick = cands[0] if cands else (next(iter(available)) if available else None)
        if pick is not None:
            assignment[idx] = available.pop(pick)
    for idx, (s1, s2) in enumerate(R32):
        t1 = pos[s1]
        t2 = assignment.get(idx) if s2.startswith('3:') else pos[s2]
        if t2 is None:  # safety net
            t2 = pos['B2']
        bracket.append(sim_knockout(t1, t2))
    # R16 -> QF -> SF -> F by sequential pairing of the official match order
    while len(bracket) > 1:
        if len(bracket) == 4:
            for t in bracket: semis[t] += 1
        if len(bracket) == 2:
            for t in bracket: finalists[t] += 1
        bracket = [sim_knockout(bracket[i], bracket[i + 1]) for i in range(0, len(bracket), 2)]
    champions[bracket[0]] += 1

print("\n=== CURRENT ELO (top 12) ===")
for t, e in sorted(cur_elo.items(), key=lambda kv: -kv[1])[:12]:
    print(f"{t:<22}{e:7.0f}")

print(f"\n=== CHAMPIONSHIP PROBABILITIES ({N_SIM:,} simulations) ===")
for t, c in sorted(champions.items(), key=lambda kv: -kv[1])[:15]:
    print(f"{t:<22}win {100*c/N_SIM:5.1f}%   final {100*finalists[t]/N_SIM:5.1f}%   semi {100*semis[t]/N_SIM:5.1f}%")
