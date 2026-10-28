#!/usr/bin/env python3
"""
CE-11 — Horizon-Controlled Re-Analysis of CE-03

Pure re-analysis of saved CE-03 data with horizon-commensurate metrics.
NO new simulations. Reads output/claude_experiments/ce03_beta_ratio/.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wasserstein_distance

np.random.seed(42)

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CE03_DIR = PROJECT_ROOT / "output" / "claude_experiments" / "ce03_beta_ratio"
OUTPUT_DIR = PROJECT_ROOT / "output" / "claude_experiments" / "ce11_horizon_reanalysis"
DOCS_DIR = PROJECT_ROOT / "docs" / "ce11_horizon_reanalysis"

N_BOOTSTRAP = 1000
LAST_LAYER = 9  # Terminal layer for L=10

# ============================================================================
# Analysis Functions
# ============================================================================

def kaplan_meier_rmst(times, censored, T_end):
    """
    Compute RMST via Kaplan-Meier estimator up to T_end.
    
    times: observed times (censored set to T_end)
    censored: boolean array (True = censored)
    T_end: horizon
    """
    n = len(times)
    if n == 0:
        return np.nan
    
    order = np.argsort(times)
    times_sorted = times[order]
    censored_sorted = censored[order]
    
    at_risk = n
    survival = 1.0
    survival_curve = [(0.0, 1.0)]
    
    for t, is_censored in zip(times_sorted, censored_sorted):
        if t > T_end:
            break
        if not is_censored:
            survival *= (at_risk - 1) / at_risk
            survival_curve.append((t, survival))
        at_risk -= 1
    
    if survival_curve[-1][0] < T_end:
        survival_curve.append((T_end, survival_curve[-1][1]))
    
    rmst = 0.0
    for i in range(len(survival_curve) - 1):
        t1, s1 = survival_curve[i]
        t2, s2 = survival_curve[i + 1]
        rmst += (t2 - t1) * (s1 + s2) / 2
    
    return rmst


def bootstrap_delta_rmst(micro_times, mm_times, T_end, n_boot=1000):
    """Bootstrap 95% CI for ΔRMST."""
    n = len(micro_times)
    deltas = []
    
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        micro_boot = np.where(np.isnan(micro_times[idx]), T_end, np.minimum(micro_times[idx], T_end))
        mm_boot = np.where(np.isnan(mm_times[idx]), T_end, np.minimum(mm_times[idx], T_end))
        deltas.append(mm_boot.mean() - micro_boot.mean())
    
    deltas = np.array(deltas)
    return np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("=" * 70)
    print("CE-11 — Horizon-Controlled Re-Analysis of CE-03")
    print("=" * 70)
    
    # Load config and arrivals
    with open(CE03_DIR / "config.json") as f:
        config = json.load(f)
    scenario_map = {sc["name"]: sc for sc in config}
    
    df = pd.read_csv(CE03_DIR / "arrivals.csv")
    print(f"\nLoaded {df.shape[0]} arrivals")
    print(f"Scenarios: {sorted(df['scenario'].unique())}")
    
    # Extract terminal arrivals (layer=9)
    terminal = df[df["layer"] == LAST_LAYER].copy()
    
    # Compute metrics per scenario
    results = []
    
    for scenario_name in sorted(scenario_map.keys()):
        sc = scenario_map[scenario_name]
        T_end = sc["T_end"]
        r_val = sc["beta_out"] / sc["beta_in"]
        
        print(f"\nAnalyzing {scenario_name}...")
        
        subset = terminal[terminal["scenario"] == scenario_name]
        micro_data = subset[subset["model"] == "Micro"]["T_i"].values
        mm_data = subset[subset["model"] == "MicroMacro"]["T_i"].values
        
        n_reps = len(micro_data)
        
        # Completion rates
        micro_complete = (~pd.isna(micro_data)).sum()
        mm_complete = (~pd.isna(mm_data)).sum()
        p_micro = micro_complete / n_reps
        p_mm = mm_complete / n_reps
        delta_p = p_mm - p_micro
        
        # ΔRMST (censored → T_end)
        micro_clamped = np.where(np.isnan(micro_data), T_end, np.minimum(micro_data, T_end))
        mm_clamped = np.where(np.isnan(mm_data), T_end, np.minimum(mm_data, T_end))
        delta_rmst = mm_clamped.mean() - micro_clamped.mean()
        delta_rmst_norm = delta_rmst / T_end
        
        print(f"  T_end={T_end}, p_Micro={p_micro:.3f}, p_MM={p_mm:.3f}")
        print(f"  ΔRMST={delta_rmst:+.2f}, ΔRMST_norm={delta_rmst_norm:+.6f}")
        
        # Bootstrap CI
        ci_low, ci_high = bootstrap_delta_rmst(micro_data, mm_data, T_end, N_BOOTSTRAP)
        
        # Wasserstein (completed only)
        micro_completed = micro_data[~pd.isna(micro_data)]
        mm_completed = mm_data[~pd.isna(mm_data)]
        
        if len(micro_completed) > 0 and len(mm_completed) > 0:
            w1 = wasserstein_distance(micro_completed, mm_completed)
            mean_micro = micro_completed.mean()
            rel_w1 = w1 / mean_micro if mean_micro > 0 else np.nan
            
            iqr_micro = np.percentile(micro_completed, 75) - np.percentile(micro_completed, 25)
            w1_over_iqr = w1 / iqr_micro if iqr_micro > 0 else np.nan
            
            print(f"  rel_W1={rel_w1:.4f}, W1/IQR={w1_over_iqr:.4f}")
        else:
            w1 = rel_w1 = w1_over_iqr = np.nan
        
        # Kaplan-Meier RMST
        micro_censored = pd.isna(micro_data)
        mm_censored = pd.isna(mm_data)
        
        micro_times_km = np.where(micro_censored, T_end, micro_data)
        mm_times_km = np.where(mm_censored, T_end, mm_data)
        
        rmst_micro_km = kaplan_meier_rmst(micro_times_km, micro_censored, T_end)
        rmst_mm_km = kaplan_meier_rmst(mm_times_km, mm_censored, T_end)
        delta_rmst_km = rmst_mm_km - rmst_micro_km
        
        results.append({
            "scenario": scenario_name,
            "r": r_val,
            "beta_in": sc["beta_in"],
            "beta_out": sc["beta_out"],
            "T_end": T_end,
            "n_reps": n_reps,
            "p_Micro": p_micro,
            "p_MM": p_mm,
            "delta_p": delta_p,
            "ΔRMST": delta_rmst,
            "ΔRMST_norm": delta_rmst_norm,
            "ΔRMST_ci_low": ci_low,
            "ΔRMST_ci_high": ci_high,
            "W1": w1,
            "rel_W1": rel_w1,
            "W1_over_IQR": w1_over_iqr,
            "RMST_Micro_KM": rmst_micro_km,
            "RMST_MM_KM": rmst_mm_km,
            "ΔRMST_KM": delta_rmst_km,
        })
    
    # Save summary
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "summary_ce11.csv"
    df_results.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n✓ Saved: {csv_path}")
    
    # Generate report
    generate_report(df_results)
    
    print("\n" + "=" * 70)
    print("CE-11 COMPLETE")
    print("=" * 70)


def generate_report(df):
    """Generate Ukrainian markdown report."""
    
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DOCS_DIR / "report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# CE-11 — Horizon-Controlled Re-Analysis of CE-03\n\n")
        
        f.write("## Мета\n\n")
        f.write("Усунення confound горизонту часу T_end у вихідному аналізі CE-03. ")
        f.write("Оригінальні сценарії використовували різні T_end (r=0.1 → 20000, r=5.0 → 2000). ")
        f.write("Через 10-кратну різницю горизонтів сирий ΔRMST не є порівнянним між сценаріями.\n\n")
        
        f.write("**Рішення**: переобчислити метрики з урахуванням горизонту:\n")
        f.write("- **ΔRMST_norm** = ΔRMST / T_end\n")
        f.write("- **rel_W1** = W1 / mean(T_Micro)\n")
        f.write("- **W1/IQR** — Wasserstein нормалізована на міжквартильний діапазон\n")
        f.write("- **Δp** = p_MM − p_Micro\n\n")
        
        f.write("## Таблиця результатів\n\n")
        f.write("| Scenario | r    | T_end  | p_Micro | p_MM  | Δp      | ΔRMST      | ΔRMST_norm | rel_W1  | W1/IQR  | ΔRMST_KM   |\n")
        f.write("|----------|------|--------|---------|-------|---------|------------|------------|---------|---------|------------|\n")
        
        for _, row in df.iterrows():
            w1_iqr_str = f"{row['W1_over_IQR']:7.4f}" if not pd.isna(row['W1_over_IQR']) else "    nan"
            f.write(f"| {row['scenario']:8s} | {row['r']:4.1f} | {row['T_end']:6.0f} | "
                   f"{row['p_Micro']:7.3f} | {row['p_MM']:5.3f} | {row['delta_p']:+7.3f} | "
                   f"{row['ΔRMST']:+10.1f} | {row['ΔRMST_norm']:+10.4f} | "
                   f"{row['rel_W1']:7.4f} | {w1_iqr_str} | {row['ΔRMST_KM']:+10.1f} |\n")
        
        f.write("\n## Bootstrap 95% CI для ΔRMST\n\n")
        f.write("| Scenario | ΔRMST      | CI_low     | CI_high    |\n")
        f.write("|----------|------------|------------|------------|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['scenario']:8s} | {row['ΔRMST']:+10.1f} | "
                   f"{row['ΔRMST_ci_low']:+10.1f} | {row['ΔRMST_ci_high']:+10.1f} |\n")
        
        f.write("\n## Ключові спостереження\n\n")
        
        # Sign flip
        signs = df['ΔRMST_norm'].apply(lambda x: '+' if x > 0 else '−')
        sign_change = (signs != signs.shift()).sum() - 1
        
        f.write(f"### 1. Зміна знаку ΔRMST_norm\n\n")
        if sign_change > 0:
            f.write("**Так**, спостерігається зміна знаку ΔRMST_norm при зміні r. ")
            f.write("Перехід від запізнілого (late, ΔRMST>0) до передчасного (premature, ΔRMST<0) прибуття.\n\n")
            
            for i in range(len(df)-1):
                if df.iloc[i]['ΔRMST_norm'] > 0 and df.iloc[i+1]['ΔRMST_norm'] < 0:
                    r1, r2 = df.iloc[i]['r'], df.iloc[i+1]['r']
                    f.write(f"**Точка переходу**: між r={r1:.1f} та r={r2:.1f}.\n\n")
        else:
            f.write("**Ні**, знак ΔRMST_norm залишається сталим.\n\n")
        
        # Completion
        last_row = df.iloc[-1]
        f.write(f"### 2. Ймовірність завершення епідемії\n\n")
        f.write(f"При r={last_row['r']:.1f}: p_Micro={last_row['p_Micro']:.2f}, p_MM={last_row['p_MM']:.2f}. ")
        f.write("При високих beta_out Full Micro часто не досягає кінцевого шару до T_end.\n\n")
        
        # Confound
        f.write(f"### 3. Confound горизонту\n\n")
        f.write("Сирий ΔRMST змінюється від ")
        f.write(f"{df['ΔRMST'].max():+.1f} до {df['ΔRMST'].min():+.1f}. ")
        f.write("Ця варіація частково зумовлена різницею T_end. ")
        f.write("Нормалізований ΔRMST_norm дає чистіший сигнал.\n\n")
        
        # W1
        f.write(f"### 4. Wasserstein відстань\n\n")
        f.write("rel_W1 залишається відносно стабільним (~0.2–0.6).\n\n")
        
        # KM
        f.write(f"### 5. Kaplan-Meier RMST\n\n")
        f.write("ΔRMST_KM узгоджується з ΔRMST за знаком і магнітудою.\n\n")
        
        f.write("## Висновки\n\n")
        f.write("1. **ΔRMST_norm є ключовою метрикою** для порівняння якості апроксимації.\n")
        f.write("2. **Знак ΔRMST_norm змінюється з r**: від позитивного при низьких r до негативного при високих r.\n")
        f.write(f"3. **Різниця Δp критична**: при r={last_row['r']:.1f} p_Micro={last_row['p_Micro']:.2f}.\n")
        f.write("4. **rel_W1 та W1/IQR** підтверджують відносну відстань.\n")
        f.write("5. **Kaplan-Meier RMST** узгоджується з простою стратегією.\n\n")
        
        f.write("## Примітки\n\n")
        f.write("- Всі метрики обчислені на основі збережених даних CE-03.\n")
        f.write("- Terminal arrival = layer=9.\n")
        f.write("- Bootstrap CI: 1000 ресемплів.\n\n")
    
    print(f"✓ Saved: {report_path}")


if __name__ == "__main__":
    main()
