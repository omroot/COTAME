# Lead/Lag Summary — MM Net Change vs Price Return (Tuesday-Tuesday)

Full-sample Spearman rank correlations between weekly MM net position change and weekly price return at various lags.
All notebooks use the **as_of_date** join (Tuesday-Tuesday alignment).

`*` = significant at 5%

| Contract | N | lag -2 | lag -1 | lag 0 | lag +1 | lag +2 |
|----------|-----|--------|--------|-------|--------|--------|
| **WTI** | 812 | -0.006 (p=8.6e-01) | +0.087 (p=1.3e-02)* | **+0.629** (p=9.8e-91)* | +0.017 (p=6.3e-01) | -0.061 (p=8.4e-02) |
| **Brent** | 801 | +0.035 (p=3.2e-01) | +0.032 (p=3.7e-01) | **+0.275** (p=2.2e-15)* | -0.042 (p=2.3e-01) | +0.037 (p=2.9e-01) |
| **HO** | 807 | -0.051 (p=1.5e-01) | **-0.252** (p=4.2e-13)* | **-0.342** (p=1.3e-23)* | -0.038 (p=2.9e-01) | +0.079 (p=2.5e-02)* |
| **RBOB** | 807 | -0.011 (p=7.5e-01) | -0.067 (p=5.7e-02) | -0.045 (p=2.1e-01) | -0.051 (p=1.5e-01) | +0.002 (p=9.6e-01) |
| **Gasoil** | 800 | -0.043 (p=2.3e-01) | +0.046 (p=2.0e-01) | +0.079 (p=2.5e-02)* | -0.027 (p=4.4e-01) | -0.099 (p=5.0e-03)* |

## Lag convention

- **Negative lag** (e.g. lag -1): price return this week vs MM change *next* week — price leads MM
- **Lag 0**: same-week (contemporaneous)
- **Positive lag** (e.g. lag +1): MM change this week vs price return *next* week — MM leads price

## Key takeaways

- **WTI** has by far the strongest contemporaneous relationship (+0.63, p~0). A weak but significant price-leads-MM signal exists at lag -1 (+0.09, p=0.013).
- **Brent** shows the same positive direction but much weaker (+0.28, p=2.2e-15), purely contemporaneous. The ICE+CFTC aggregation likely adds noise.
- **HO** is the outlier — the relationship is **negative** at lag 0 (-0.34) and lag -1 (-0.25), both highly significant. Price moves precede and oppose MM positioning changes (contrarian / mean-reversion behavior).
- **RBOB** shows no statistically significant relationship at any lag.
- **Gasoil** is marginal — weak positive contemporaneous (+0.08, p=0.025) and a negative lag +2 (-0.10, p=0.005), but the signal is unstable across regimes (flips sign in 2-year blocks).
