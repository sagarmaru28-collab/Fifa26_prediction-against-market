"""
Stage 3: Post-Round-of-16 forecast (run 2026-07-08)
Elo + form updated through all results to July 7, then Monte Carlo
from the actual quarterfinal bracket.

QF bracket:  France-Morocco | Spain-Belgium    (same side)
             Norway-England | Argentina-Switzerland

Output at time of run (50,000 sims):
  Argentina 27.8% | France 22.0% | Spain 21.8% | England 12.0%
  Morocco 5.3% | Belgium 5.0% | Norway 3.6% | Switzerland 2.4%
Market at same time: France ~33%, Spain ~18%, Argentina ~17%, England ~8%
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.linear_model import PoissonRegressor

rng = np.random.default_rng(42)

df = pd.read_csv('results.csv', parse_dates=['date'])
df = df.dropna(subset=['home_score', 'away_score']).sort_values('date').reset_index(drop=True)

def k_factor(t):
    t = str(t)
    if 'FIFA World Cup' in t and 'qualification' not in t: return 60
    if 'qualification' in t or 'UEFA Euro' in t or 'Copa' in t or 'Africa Cup' in t or 'AFC Asian Cup' in t or 'Gold Cup' in t: return 40
    if 'Friendly' in t: return 20
    return 30

elo = defaultdict(lambda: 1500.0)
HOME_ADV = 80.0
rgf = defaultdict(list); rga = defaultdict(list); rows = []
for r in df.itertuples():
    h, a = r.home_team, r.away_team
    eh, ea = elo[h], elo[a]
    hb = 0.0 if r.neutral else HOME_ADV
    exp = 1 / (1 + 10 ** (-((eh + hb) - ea) / 400))
    hs, as_ = int(r.home_score), int(r.away_score)
    res = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
    rows.append({'date': r.date, 'hs': hs, 'as': as_,
        'elo_diff': (eh + hb) - ea, 'neutral': int(r.neutral),
        'form_gf_h': np.mean(rgf[h][-10:]) if rgf[h] else 1.3,
        'form_ga_h': np.mean(rga[h][-10:]) if rga[h] else 1.3,
        'form_gf_a': np.mean(rgf[a][-10:]) if rgf[a] else 1.3,
        'form_ga_a': np.mean(rga[a][-10:]) if rga[a] else 1.3})
    K = k_factor(r.tournament); m = abs(hs - as_)
    mov = 1.0 if m <= 1 else (1.5 if m == 2 else (11 + m) / 8.0)
    d = K * mov * (res - exp)
    elo[h] += d; elo[a] -= d
    rgf[h].append(hs); rga[h].append(as_)
    rgf[a].append(as_); rga[a].append(hs)

md = pd.DataFrame(rows)
tr = md[md.date >= '1990-01-01'].copy()
tr['elo_diff_s'] = tr['elo_diff'] / 400
GF = ['elo_diff_s', 'neutral', 'form_gf_h', 'form_ga_a']
GA = ['elo_diff_s', 'neutral', 'form_gf_a', 'form_ga_h']
ph = PoissonRegressor(alpha=1e-4, max_iter=2000).fit(tr[GF], tr['hs'])
pa = PoissonRegressor(alpha=1e-4, max_iter=2000).fit(tr[GA], tr['as'])

HOSTS = {'United States', 'Canada', 'Mexico'}
def gf_(t): return np.mean(rgf[t][-10:])
def ga_(t): return np.mean(rga[t][-10:])
LAM = {}
def lam(t1, t2):
    if (t1, t2) not in LAM:
        bonus = HOME_ADV if t1 in HOSTS else (-HOME_ADV if t2 in HOSTS else 0.0)
        ed = elo[t1] - elo[t2] + bonus
        nt = 0 if (t1 in HOSTS or t2 in HOSTS) else 1
        xh = ph.predict(pd.DataFrame([{'elo_diff_s': ed/400, 'neutral': nt, 'form_gf_h': gf_(t1), 'form_ga_a': ga_(t2)}]))[0]
        xa = pa.predict(pd.DataFrame([{'elo_diff_s': ed/400, 'neutral': nt, 'form_gf_a': gf_(t2), 'form_ga_h': ga_(t1)}]))[0]
        LAM[(t1, t2)] = (max(xh, .05), max(xa, .05))
    return LAM[(t1, t2)]
def ko(t1, t2):
    l1, l2 = lam(t1, t2)
    g1, g2 = rng.poisson(l1), rng.poisson(l2)
    if g1 != g2: return t1 if g1 > g2 else t2
    p = 1 / (1 + 10 ** (-(elo[t1] - elo[t2]) / 400)); p = .5 + .5 * (p - .5)
    return t1 if rng.random() < p else t2

QF = [('France', 'Morocco'), ('Spain', 'Belgium'), ('Norway', 'England'), ('Argentina', 'Switzerland')]

print("=== QUARTERFINAL WIN PROBABILITIES ===")
for t1, t2 in QF:
    w = sum(ko(t1, t2) == t1 for _ in range(20000))
    print(f"{t1} vs {t2}: {t1} {w/200:.0f}% - {t2} {100 - w/200:.0f}%")

N = 50000
champ = defaultdict(int); fin = defaultdict(int)
for _ in range(N):
    sf1 = ko(*QF[0]); sf2 = ko(*QF[1]); sf3 = ko(*QF[2]); sf4 = ko(*QF[3])
    f1 = ko(sf1, sf2); f2 = ko(sf3, sf4)
    fin[f1] += 1; fin[f2] += 1
    champ[ko(f1, f2)] += 1

print(f"\n=== TITLE PROBABILITIES ({N:,} sims from QF) ===")
for t, c in sorted(champ.items(), key=lambda kv: -kv[1]):
    print(f"{t:<14}win {100*c/N:5.1f}%   reach final {100*fin[t]/N:5.1f}%")
