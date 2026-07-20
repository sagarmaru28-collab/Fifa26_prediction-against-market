"""
2026 FIFA World Cup — POST-GROUP-STAGE forecast
Continuation of the pre-tournament model from this chat.

Changes vs the pre-tournament version:
  - Elo + rolling form now updated through ALL 72 group-stage results (to 2026-06-27)
  - No group simulation: the ACTUAL Round of 32 bracket is seeded from real fixtures
  - Monte Carlo runs the knockout bracket only (R32 -> R16 -> QF -> SF -> Final)
The match/goal models themselves are identical (Poisson goals, scaled Elo diff).
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
recent_gf = defaultdict(list); recent_ga = defaultdict(list)
rows = []
for r in df.itertuples():
    h, a = r.home_team, r.away_team
    eh, ea = elo[h], elo[a]
    hb = 0.0 if r.neutral else HOME_ADV
    exp_h = 1/(1+10**(-((eh+hb)-ea)/400))
    hs, as_ = int(r.home_score), int(r.away_score)
    res = 1.0 if hs>as_ else (0.5 if hs==as_ else 0.0)
    fN=10
    rows.append({'date':r.date,'hs':hs,'as':as_,
        'elo_diff':(eh+hb)-ea,'neutral':int(r.neutral),
        'form_gf_h':np.mean(recent_gf[h][-fN:]) if recent_gf[h] else 1.3,
        'form_ga_h':np.mean(recent_ga[h][-fN:]) if recent_ga[h] else 1.3,
        'form_gf_a':np.mean(recent_gf[a][-fN:]) if recent_gf[a] else 1.3,
        'form_ga_a':np.mean(recent_ga[a][-fN:]) if recent_ga[a] else 1.3})
    K=k_factor(r.tournament); margin=abs(hs-as_)
    mov = 1.0 if margin<=1 else (1.5 if margin==2 else (11+margin)/8.0)
    d = K*mov*(res-exp_h)
    elo[h]+=d; elo[a]-=d
    recent_gf[h].append(hs); recent_ga[h].append(as_)
    recent_gf[a].append(as_); recent_ga[a].append(hs)

md = pd.DataFrame(rows)
train = md[md.date>='1990-01-01'].copy()
train['elo_diff_s'] = train['elo_diff']/400.0
GF=['elo_diff_s','neutral','form_gf_h','form_ga_a']
GA=['elo_diff_s','neutral','form_gf_a','form_ga_h']
ph=PoissonRegressor(alpha=1e-4,max_iter=2000).fit(train[GF],train['hs'])
pa=PoissonRegressor(alpha=1e-4,max_iter=2000).fit(train[GA],train['as'])

# Elo/form snapshot now reflects everything through 2026-06-27 (group stage complete)
HOSTS={'United States','Canada','Mexico'}
def gf_(t): return np.mean(recent_gf[t][-10:]) if recent_gf[t] else 1.3
def ga_(t): return np.mean(recent_ga[t][-10:]) if recent_ga[t] else 1.3

# ACTUAL Round of 32 bracket (order = official knockout bracket pairing)
R32 = [
    ('South Africa','Canada'),
    ('Brazil','Japan'),
    ('Germany','Paraguay'),
    ('Netherlands','Morocco'),
    ('Ivory Coast','Norway'),
    ('France','Sweden'),
    ('Mexico','Ecuador'),
    ('England','DR Congo'),
    ('Belgium','Senegal'),
    ('United States','Bosnia and Herzegovina'),
    ('Spain','Austria'),
    ('Portugal','Croatia'),
    ('Switzerland','Algeria'),
    ('Australia','Egypt'),
    ('Argentina','Cape Verde'),
    ('Colombia','Ghana'),
]

LAM={}
def lambdas(t1,t2):
    if (t1,t2) in LAM: return LAM[(t1,t2)]
    bonus = HOME_ADV if t1 in HOSTS else (-HOME_ADV if t2 in HOSTS else 0.0)
    ed = elo[t1]-elo[t2]+bonus
    neutral = 0 if (t1 in HOSTS or t2 in HOSTS) else 1
    xh=ph.predict(pd.DataFrame([{'elo_diff_s':ed/400,'neutral':neutral,'form_gf_h':gf_(t1),'form_ga_a':ga_(t2)}]))[0]
    xa=pa.predict(pd.DataFrame([{'elo_diff_s':ed/400,'neutral':neutral,'form_gf_a':gf_(t2),'form_ga_h':ga_(t1)}]))[0]
    LAM[(t1,t2)]=(max(xh,0.05),max(xa,0.05)); return LAM[(t1,t2)]

def ko(t1,t2):
    l1,l2=lambdas(t1,t2)
    g1,g2=rng.poisson(l1),rng.poisson(l2)
    if g1!=g2: return t1 if g1>g2 else t2
    p1=1/(1+10**(-(elo[t1]-elo[t2])/400)); p1=0.5+0.5*(p1-0.5)
    return t1 if rng.random()<p1 else t2

N=20000
champ=defaultdict(int); fin=defaultdict(int); semi=defaultdict(int); quart=defaultdict(int)
for _ in range(N):
    b=[ko(a,b_) for a,b_ in R32]   # R16 (16 teams)
    for t in b: quart[t]+=1
    while len(b)>1:
        if len(b)==4:
            for t in b: semi[t]+=1
        if len(b)==2:
            for t in b: fin[t]+=1
        b=[ko(b[i],b[i+1]) for i in range(0,len(b),2)]
    champ[b[0]]+=1

print("=== ELO AFTER GROUP STAGE (top 12) ===")
teams=set(t for p in R32 for t in p)
for t,e in sorted(((t,elo[t]) for t in teams), key=lambda kv:-kv[1])[:12]:
    print(f"{t:<24}{e:7.0f}")

print(f"\n=== TITLE PROBABILITIES POST-GROUP-STAGE ({N:,} knockout sims) ===")
for t,c in sorted(champ.items(),key=lambda kv:-kv[1]):
    if c/N>=0.005:
        print(f"{t:<24}win {100*c/N:5.1f}%  reach final {100*fin[t]/N:5.1f}%  reach SF {100*semi[t]/N:5.1f}%")
