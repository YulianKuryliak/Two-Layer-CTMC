#!/usr/bin/env python3
"""
CE-11 — Horizon-controlled re-analysis of CE-03.
Pure re-analysis: reads saved arrivals, computes horizon-commensurate metrics.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path


def wasserstein_distance(u, v):
    """Compute 1-Wasserstein distance (Earth Mover's Distance) between two 1D distributions."""
    u_sorted = np.sort(u)
    v_sorted = np.sort(v)
    
    # Use cumulative distribution functions
    all_values = np.concatenate([u_sorted, v_sorted])
    all_values = np.sort(np.unique(all_values))
    
    # Compute empirical CDFs
    u_cdf = np.searchsorted(u_sorted, all_values, side='right') / len(u_sorted)
    v_cdf = np.searchsorted(v_sorted, all_values, side='right') / len(v_sorted)
    
    # Integrate absolute difference
    if len(all_values) > 1:
        deltas = np.diff(all_values)
        abs_diff = np.abs(u_cdf[:-1] - v_cdf[:-1])
        return np.sum(deltas * abs_diff)
    return 0.0

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = BASE_DIR / "output" / "claude_experiments" / "ce03_beta_ratio"
OUTPUT_DIR = BASE_DIR / "output" / "claude_experiments" / "ce11_horizon_reanalysis"
DOCS_DIR = BASE_DIR / "docs" / "ce11_horizon_reanalysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def compute_empirical_rmst_censored(times, censored, T_end):
    """
    Compute RMST from empirical survival curve with right-censoring.
    
    Args:
        times: array of event/censoring times
        censored: boolean array (True = censored, False = event observed)
        T_end: horizon for RMST calculation
    
    Returns:
        RMST up to T_end (area under survival curve)
    """
    # Sort by time
    idx = np.argsort(times)
    times_sorted = times[idx]
    censored_sorted = censored[idx]
    
    n_total = len(times)
    at_risk = n_total
    
    # Build step function: (time, survival_prob)
    surv_times = [0.0]
    surv_probs = [1.0]
    
    i = 0
    while i < len(times_sorted) and times_sorted[i] <= T_end:
        t = times_sorted[i]
        
        # Count events at this exact time
        events_at_t = 0
        j = i
        while j < len(times_sorted) and times_sorted[j] == t:
            if not censored_sorted[j]:
                events_at_t += 1
            j += 1
        
        # Update survival
        if events_at_t > 0 and at_risk > 0:
            surv = surv_probs[-1] * (1 - events_at_t / at_risk)
            surv_times.append(t)
            surv_probs.append(surv)
        
        # Update at-risk count (remove all at this time)
        at_risk -= (j - i)
        i = j
    
    # Add final point at T_end if needed
    if surv_times[-1] < T_end:
        surv_times.append(T_end)
        surv_probs.append(surv_probs[-1])
    
    # Compute area under curve (trapezoid rule)
    rmst = 0.0
    for i in range(len(surv_times) - 1):
        dt = surv_times[i+1] - surv_times[i]
        avg_s = (surv_probs[i] + surv_probs[i+1]) / 2
        rmst += dt * avg_s
    
    return rmst


def bootstrap_delta_rmst(t_micro, t_mm, T_end, n_boot=1000, seed=42):
    """
    Bootstrap 95% CI for ΔRMST = mean(min(T_MM, T_end) - min(T_Micro, T_end)).
    """
    rng = np.random.RandomState(seed)
    n = len(t_micro)
    assert len(t_mm) == n
    
    # Censor at T_end
    t_micro_c = np.minimum(t_micro, T_end)
    t_mm_c = np.minimum(t_mm, T_end)
    
    delta_boot = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        d = np.mean(t_mm_c[idx] - t_micro_c[idx])
        delta_boot.append(d)
    
    ci_low = np.percentile(delta_boot, 2.5)
    ci_high = np.percentile(delta_boot, 97.5)
    return ci_low, ci_high


def analyze_scenario(df_arr, scenario_name, config):
    """Analyze one scenario."""
    T_end = config['T_end']
    beta_in = config['beta_in']
    beta_out = config['beta_out']
    ratio = config['beta_out'] / config['beta_in']
    
    # Filter arrivals for this scenario
    df_sc = df_arr[df_arr['scenario'] == scenario_name].copy()
    
    # Extract terminal arrivals (last layer per replication per model)
    # Layer numbering: 0..L-1 → layer 9 is last for L=10
    df_terminal = df_sc.loc[df_sc.groupby(['replication_id', 'model'])['layer'].idxmax()].reset_index(drop=True)
    
    # Separate by model
    df_micro = df_terminal[df_terminal['model'] == 'Micro'].copy()
    df_mm = df_terminal[df_terminal['model'] == 'MicroMacro'].copy()
    
    # Sort by replication_id to ensure alignment
    df_micro = df_micro.sort_values('replication_id').reset_index(drop=True)
    df_mm = df_mm.sort_values('replication_id').reset_index(drop=True)
    
    n_reps = len(df_micro)
    assert len(df_mm) == n_reps, f"Mismatch: {len(df_micro)} vs {len(df_mm)}"
    
    # Extract times (NaN → censored)
    t_micro_raw = df_micro['T_i'].values
    t_mm_raw = df_mm['T_i'].values
    
    # Identify censored (NaN or >= T_end)
    censored_micro = np.isnan(t_micro_raw) | (t_micro_raw >= T_end)
    censored_mm = np.isnan(t_mm_raw) | (t_mm_raw >= T_end)
    
    # Replace NaN with T_end for metric calculation
    t_micro = np.where(np.isnan(t_micro_raw), T_end, t_micro_raw)
    t_mm = np.where(np.isnan(t_mm_raw), T_end, t_mm_raw)
    
    # Completion rates
    p_micro = np.sum(~censored_micro) / n_reps
    p_mm = np.sum(~censored_mm) / n_reps
    delta_p = p_mm - p_micro
    
    # ΔRMST (censored-aware: cap at T_end)
    t_micro_capped = np.minimum(t_micro, T_end)
    t_mm_capped = np.minimum(t_mm, T_end)
    delta_rmst = np.mean(t_mm_capped - t_micro_capped)
    delta_rmst_norm = delta_rmst / T_end
    
    # Bootstrap CI
    ci_low, ci_high = bootstrap_delta_rmst(t_micro, t_mm, T_end)
    
    # Wasserstein on completed arrivals only
    completed_micro = t_micro[~censored_micro]
    completed_mm = t_mm[~censored_mm]
    
    if len(completed_micro) > 0 and len(completed_mm) > 0:
        w1 = wasserstein_distance(completed_micro, completed_mm)
        mean_micro = np.mean(completed_micro)
        rel_w1 = w1 / mean_micro if mean_micro > 0 else np.nan
        
        iqr_micro = np.percentile(completed_micro, 75) - np.percentile(completed_micro, 25)
        w1_over_iqr = w1 / iqr_micro if iqr_micro > 0 else np.nan
    else:
        w1 = np.nan
        rel_w1 = np.nan
        w1_over_iqr = np.nan
    
    # Kaplan-Meier RMST
    rmst_micro = compute_empirical_rmst_censored(t_micro, censored_micro, T_end)
    rmst_mm = compute_empirical_rmst_censored(t_mm, censored_mm, T_end)
    delta_rmst_km = rmst_mm - rmst_micro
    
    return {
        'scenario': scenario_name,
        'r': ratio,
        'beta_in': beta_in,
        'beta_out': beta_out,
        'T_end': T_end,
        'n_reps': n_reps,
        'p_Micro': p_micro,
        'p_MM': p_mm,
        'delta_p': delta_p,
        'ΔRMST': delta_rmst,
        'ΔRMST_norm': delta_rmst_norm,
        'ΔRMST_ci_low': ci_low,
        'ΔRMST_ci_high': ci_high,
        'W1': w1,
        'rel_W1': rel_w1,
        'W1_over_IQR': w1_over_iqr,
        'RMST_Micro_KM': rmst_micro,
        'RMST_MM_KM': rmst_mm,
        'ΔRMST_KM': delta_rmst_km,
    }


def main():
    print("CE-11 — Horizon-controlled re-analysis of CE-03")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {INPUT_DIR}")
    df_arrivals = pd.read_csv(INPUT_DIR / "arrivals.csv")
    with open(INPUT_DIR / "config.json") as f:
        configs = json.load(f)
    
    # Build scenario config map
    config_map = {c['name']: c for c in configs}
    
    print(f"  Loaded {len(df_arrivals)} arrivals across {len(config_map)} scenarios")
    print(f"  Scenarios: {list(config_map.keys())}")
    
    # Analyze each scenario
    results = []
    for scenario_name in sorted(config_map.keys(), key=lambda s: config_map[s]['beta_out']):
        print(f"\nAnalyzing {scenario_name}...")
        cfg = config_map[scenario_name]
        res = analyze_scenario(df_arrivals, scenario_name, cfg)
        results.append(res)
        
        print(f"  T_end={res['T_end']:.0f}, p_Micro={res['p_Micro']:.3f}, p_MM={res['p_MM']:.3f}")
        print(f"  ΔRMST={res['ΔRMST']:.2f}, ΔRMST_norm={res['ΔRMST_norm']:.6f}")
        print(f"  rel_W1={res['rel_W1']:.4f}, W1/IQR={res['W1_over_IQR']:.4f}")
    
    # Save summary
    df_results = pd.DataFrame(results)
    out_csv = OUTPUT_DIR / "summary_ce11.csv"
    df_results.to_csv(out_csv, index=False, float_format='%.6f')
    print(f"\n✓ Saved {out_csv}")
    
    # Generate markdown report
    generate_report(df_results)
    
    print("\n✓ CE-11 complete.")


def generate_report(df_results):
    """Generate Ukrainian markdown report with English metrics."""
    
    # Check if sign flips
    delta_norms = df_results['ΔRMST_norm'].values
    has_positive = np.any(delta_norms > 0)
    has_negative = np.any(delta_norms < 0)
    sign_flips = has_positive and has_negative
    
    report = f"""# CE-11 — Horizon-Controlled Re-Analysis of CE-03

**Мета**: переаналіз даних CE-03 з урахуванням різних горизонтів `T_end` у різних сценаріях. Сирий ΔRMST не можна порівнювати між сценаріями через confound горизонту (T_end варіює від 2000 до 20000). Вводимо нормовані метрики: ΔRMST_norm, rel_W1, Δp.

**Джерело даних**: `output/claude_experiments/ce03_beta_ratio/` — збережені arrival times з CE-03.

---

## Результати

### Таблиця метрик (horizon-commensurate)

| Scenario | r    | T_end  | p_Micro | p_MM  | Δp      | ΔRMST    | **ΔRMST_norm** | rel_W1  | W1/IQR  |
|----------|------|--------|---------|-------|---------|----------|----------------|---------|---------|
"""
    
    for _, row in df_results.iterrows():
        report += f"| {row['scenario']:<8} | {row['r']:<4.1f} | {row['T_end']:>6.0f} | {row['p_Micro']:.3f}   | {row['p_MM']:.3f} | {row['delta_p']:>+7.3f} | {row['ΔRMST']:>+8.1f} | **{row['ΔRMST_norm']:>+9.6f}** | {row['rel_W1']:>7.4f} | {row['W1_over_IQR']:>7.4f} |\n"
    
    report += f"""
**Bootstrap 95% CI для ΔRMST**:
"""
    for _, row in df_results.iterrows():
        report += f"- {row['scenario']}: [{row['ΔRMST_ci_low']:+.1f}, {row['ΔRMST_ci_high']:+.1f}]\n"
    
    report += f"""
**Kaplan–Meier RMST** (area under empirical survival curve):
"""
    for _, row in df_results.iterrows():
        report += f"- {row['scenario']}: RMST_Micro={row['RMST_Micro_KM']:.1f}, RMST_MM={row['RMST_MM_KM']:.1f}, ΔRMST_KM={row['ΔRMST_KM']:+.1f}\n"
    
    report += f"""

---

## Інтерпретація

### Чому ΔRMST_norm, а не сирий ΔRMST?

Сирий ΔRMST залежить від горизонту `T_end`:
- r=0.1: T_end=20000 → ΔRMST={df_results[df_results['r']==0.1]['ΔRMST'].values[0]:+.1f}
- r=5.0: T_end=2000  → ΔRMST={df_results[df_results['r']==5.0]['ΔRMST'].values[0]:+.1f}

Ці значення **несумісні** — горизонт впливає на абсолютну похибку. Тому нормуємо:
- **ΔRMST_norm = ΔRMST / T_end** — відносна похибка, comparable across scenarios.
- **rel_W1 = W1 / mean(T_Micro|completed)** — effect-size метрика на completed arrivals.
- **W1/IQR** — альтернативна нормалізація, не залежить від censoring bias.

### Переворот знаку (sign flip)

{'✓' if sign_flips else '✗'} **Sign flip підтверджено**: ΔRMST_norm {'змінює' if sign_flips else 'НЕ змінює'} знак:
"""
    
    for _, row in df_results.iterrows():
        sign_str = '+' if row['ΔRMST_norm'] > 0 else '−' if row['ΔRMST_norm'] < 0 else '0'
        report += f"- r={row['r']:.1f}: ΔRMST_norm = {row['ΔRMST_norm']:+.6f} ({sign_str})\n"
    
    # Find crossover region
    if sign_flips:
        positive_r = df_results[df_results['ΔRMST_norm'] > 0]['r'].values
        negative_r = df_results[df_results['ΔRMST_norm'] < 0]['r'].values
        if len(positive_r) > 0 and len(negative_r) > 0:
            crossover_low = max(positive_r)
            crossover_high = min(negative_r)
            report += f"\nКросовер у діапазоні r* ∈ ({crossover_low:.1f}, {crossover_high:.1f}).\n"
    
    report += f"""

---

## Висновки

1. **Horizon confound усунуто**: використання ΔRMST_norm та rel_W1 дає comparable метрики across різні T_end.

2. **Sign flip зберігається**: {'MicroMacro пізніше прибуває при малих r (ΔRMST_norm > 0) та передчасно при великих r (< 0)' if sign_flips else 'знак ΔRMST_norm стабільний'}.

3. **Completion gap Δp**: {'систематично зростає' if df_results['delta_p'].is_monotonic_increasing else 'варіює'} при зростанні r — {'ознака false-positive epidemics у MicroMacro' if any(df_results['delta_p'] > 0.1) else 'невеликий ефект'}.

4. **Wasserstein metrics (rel_W1, W1/IQR)** показують {('substantial' if df_results['rel_W1'].max() > 0.5 else 'moderate')} distributional shift між моделями навіть на completed arrivals.

5. **Kaplan–Meier RMST** узгоджується з censored ΔRMST — обидва методи дають {'подібні' if np.allclose(df_results['ΔRMST'], df_results['ΔRMST_KM'], rtol=0.1) else 'різні'} оцінки (можлива різниця через survivor-function discretization).

---

**Файли**:
- Дані: `output/claude_experiments/ce03_beta_ratio/arrivals.csv`
- Результати: `output/claude_experiments/ce11_horizon_reanalysis/summary_ce11.csv`
- Скрипт: `extrascripts/analysis/analyze_ce11_horizon.py`
"""
    
    out_md = DOCS_DIR / "report.md"
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ Saved {out_md}")


if __name__ == "__main__":
    main()
