import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import stats

CURRENCIES = ["USD", "EUR", "GBP"]
DAYS_BACK = 365
OUT_DIR = "outputs"
CSV_PATH = os.path.join(OUT_DIR, "l1_nbu_exchange_rates_1y.csv")
COLORS = {"USD": "#2196F3", "EUR": "#4CAF50", "GBP": "#FF9800"}
ALPHA = 0.05
SEED = 42
NOISE_MODEL = "ar1"


def main():
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_or_fetch_data(CSV_PATH, CURRENCIES, DAYS_BACK)

    print("============= Оцінка динаміки тренду реальних даних =============")
    x = np.arange(len(df))
    trends = {}
    for cur in CURRENCIES:
        trends[cur] = fit_trend(x, df[cur].values)
        print_trend_info(cur, trends[cur])

    print("============= Статистичні характеристики реальних даних =============")
    all_stats = {}
    for cur in CURRENCIES:
        all_stats[cur] = compute_statistics(df[cur].values, f"{cur}_real")
        print_statistics(all_stats[cur])


    print("============= Синтез та верифікація моделі даних =============")
    synthetics = {}
    for cur in CURRENCIES:
        y_real = df[cur].values
        y_synth = synthesize_model(y_real, trends[cur], dist=NOISE_MODEL)
        synthetics[cur] = y_synth

        st_synth = compute_statistics(y_synth, f"{cur}_synth")
        print_statistics(st_synth)
        verify_model(y_real, y_synth, trends[cur], all_stats[cur], st_synth, cur)


    print("============= Аналіз отриманих результатів =============")
    for cur in CURRENCIES:
        tr, st = trends[cur], all_stats[cur]
        resid = df[cur].values - tr["y_best"]
        direction = "зростаючий" if tr["slope_lin"] > 0 else "спадаючий"
        cv_label = "висока" if st["cv_pct"] > 15 else "помірна/низька"
        skew_label = "правостороння" if st["skewness"] > 0 else "лівостороння"
        print(f"    ------------------{cur}------------------")
        print(f"    N={st['n']}, mu={st['mean']:.3f} UAH, sigma={st['std']:.3f} UAH")
        print(f"    Коефіцієнт варіації: {st['cv_pct']:.2f}% ({cv_label} мінливість)")
        print(f"    Асиметрія: {st['skewness']:.3f} ({skew_label})")
        print(f"    Загальний напрямок (лінійний нахил {tr['slope_lin']:.5f} UAH/день, "
              f"p={tr['p_slope']:.2e}): {direction}")
        print(f"    Обраний тренд: {tr['best']} "
              f"(R2={tr['r2_best']:.4f}, adj.R2={tr['adj_r2_best']:.4f}; "
              f"F-тест квадр. vs лін.: p={tr['p_f']:.4f})")
        print(f"    Залишки: std={np.std(resid, ddof=1):.3f}, "
              f"автокореляція lag-1={lag1_autocorr(resid):.3f}")
    print()

    plot_currency_trends(df, trends, f"{OUT_DIR}/l1_currency_trends.png")
    plot_residual_analysis(df, trends, f"{OUT_DIR}/l1_residual_analysis.png")
    plot_real_vs_synthetic_series(df, synthetics, f"{OUT_DIR}/l1_real_vs_synthetic_series.png")
    plot_raw_value_distributions(df, synthetics, f"{OUT_DIR}/l1_raw_distributions.png")
    plot_residual_distributions(df, trends, synthetics, f"{OUT_DIR}/l1_residual_distributions.png")
    plot_qq_normality_check(df, trends, f"{OUT_DIR}/l1_qq_normality_check.png")


    


# data parsing

def fetch_nbu_rate(currency: str, date: datetime):
    date_str = date.strftime("%Y%m%d")
    url = f"https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange?valcode={currency}&date={date_str}&json"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data:
            rate = data[0]["rate"]
            return rate
    except Exception:
        pass


def parse_nbu_data(currencies: list[str], days_back: int):
    today = datetime.today()
    dates = [today - timedelta(days=i) for i in range(days_back, -1, -1)]
    records = []
    for d in dates:
        row = {"date": d.date()}
        for cur in currencies:
            row[cur] = fetch_nbu_rate(cur, d)
        records.append(row)
    df = pd.DataFrame(records)
    df.dropna(inplace=True)

    df.reset_index(drop=True, inplace=True)
    print("-" * 64)
    print(f"Rows: {len(df)}")
    print(f"Dates: {df["date"].iloc[0]} - {df["date"].iloc[-1]}")
    for cur in currencies:
        print(f"{cur}: min={df[cur].min()}, max={df[cur].max()}, mean={df[cur].mean()}")
    print("-" * 64)

    return df


def save_to_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"File: {path}")
    print(f"Dataset size: {df.shape[0]} rows * {df.shape[1]} columns")
    return df


def load_or_fetch_data(csv_path: str, currencies: list, days_back: int):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = parse_nbu_data(currencies, days_back)
        save_to_csv(df, csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# statistics

def compute_statistics(series: np.ndarray, label: str) -> dict:
    n = len(series)
    mu = np.mean(series)
    std = np.std(series, ddof=1)
    ci = stats.t.interval(0.95, df=n - 1, loc=mu, scale=stats.sem(series))
    return dict(
        label=label, n=n, mean=mu, median=np.median(series),
        variance=np.var(series, ddof=1), std=std,
        skewness=stats.skew(series), kurtosis=stats.kurtosis(series),
        cv_pct=std / mu * 100 if mu != 0 else np.nan,
        ci95_lo=ci[0], ci95_hi=ci[1],
    )


def print_statistics(s: dict):
    print(f"""  ------------------{s['label']}------------------
    Кількість спостережень: {s['n']}
    Математичне очікування: {s['mean']:.3f}
    Медіана:                {s['median']:.3f}
    Дисперсія:              {s['variance']:.3f}
    Середньокв. відхилення: {s['std']:.3f}
    Коефіцієнт варіації, %: {s['cv_pct']:.3f}
    Асиметрія:              {s['skewness']:.3f}
    Ексцес:                 {s['kurtosis']:.3f}
    ДІ 95% (mean):          {s['ci95_lo']:.3f} - {s['ci95_hi']:.3f}
    """)


def lag1_autocorr(r: np.ndarray):
    r = r - np.mean(r)
    return float(np.sum(r[1:] * r[:-1]) / np.sum(r ** 2))


def durbin_watson(r: np.ndarray):
    return float(np.sum(np.diff(r) ** 2) / np.sum(r ** 2))


# trend

def _r2(y, y_hat):
    return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)


def fit_trend(x: np.ndarray, y: np.ndarray):
    n = len(y)
    p1 = np.polyfit(x, y, 1)
    y_lin = np.polyval(p1, x)
    r2_lin = _r2(y, y_lin)

    p2 = np.polyfit(x, y, 2)
    y_quad = np.polyval(p2, x)
    r2_quad = _r2(y, y_quad)

    adj_lin = 1 - (1 - r2_lin) * (n - 1) / (n - 2)
    adj_quad = 1 - (1 - r2_quad) * (n - 1) / (n - 3)

    slope_res = stats.linregress(x, y)

    rss1 = np.sum((y - y_lin) ** 2)
    rss2 = np.sum((y - y_quad) ** 2)
    f_stat = (rss1 - rss2) / (rss2 / (n - 3))
    p_f = stats.f.sf(f_stat, 1, n - 3)

    if p_f < ALPHA:
        best, p_best, y_best, r2_best, adj_best = "quad", p2, y_quad, r2_quad, adj_quad
    else:
        best, p_best, y_best, r2_best, adj_best = "linear", p1, y_lin, r2_lin, adj_lin

    return dict(
        p_linear=p1, y_linear=y_lin, r2_linear=r2_lin, adj_r2_linear=adj_lin,
        p_quad=p2, y_quad=y_quad, r2_quad=r2_quad, adj_r2_quad=adj_quad,
        slope_lin=slope_res.slope, p_slope=slope_res.pvalue,
        f_stat=f_stat, p_f=p_f,
        best=best, p_best=p_best, y_best=y_best, r2_best=r2_best, adj_r2_best=adj_best,
    )


def print_trend_info(cur: str, tr: dict):
    p1, p2 = tr["p_linear"], tr["p_quad"]
    print(f"""  ------------------{cur}------------------
    Лінійний:       y = {p1[0]:.6f} * x + {p1[1]:.6f}
                    R2 = {tr['r2_linear']:.6f}, adj.R2 = {tr['adj_r2_linear']:.6f}
                    значущість нахилу: p = {tr['p_slope']:.2e}
    Квадратичний:   y = {p2[0]:.8f} * x^2 + {p2[1]:.6f} * x + {p2[2]:.6f}
                    R2 = {tr['r2_quad']:.6f}, adj.R2 = {tr['adj_r2_quad']:.6f}
    F-тест (квадр. vs лін.): F = {tr['f_stat']:.3f}, p = {tr['p_f']:.4f}
    Кращий тренд: {tr['best']}
    """)


# Synthesis and verification

def synthesize_model(y_real: np.ndarray, tr: dict, dist: str = "ar1"):
    n = len(y_real)
    trend = tr["y_best"]
    resid = y_real - trend
    mu_r = np.mean(resid)
    sigma = np.std(resid, ddof=1)

    if dist == "normal":
        noise = np.random.normal(mu_r, sigma, size=n)
    elif dist == "uniform":
        a, b = np.min(resid), np.max(resid)
        noise = np.random.uniform(a, b, size=n)
    elif dist == "ar1":
        phi = np.clip(lag1_autocorr(resid), -0.99, 0.99)
        sigma_eps = sigma * np.sqrt(1 - phi ** 2)
        noise = np.empty(n)
        noise[0] = np.random.normal(0, sigma)
        for t in range(1, n):
            noise[t] = phi * noise[t - 1] + np.random.normal(0, sigma_eps)
        noise += mu_r
    else:
        raise ValueError(f"Unknown noise model: {dist}")
    return trend + noise


def verify_model(y_real, y_synth, tr, st_real, st_synth, cur):
    x = np.arange(len(y_real))
    d, p = stats.ks_2samp(y_real, y_synth)
    verdict = "розподіли не відрізняються" if p > ALPHA else "значуща відмінність"
    print(f"    KS-test значень   [{cur}]: D={d:.4f} p={p:.4f} - {verdict}")

    r_real = y_real - tr["y_best"]
    r_synth = y_synth - tr["y_best"]
    d, p = stats.ks_2samp(r_real, r_synth)
    verdict = "розподіли не відрізняються" if p > ALPHA else "значуща відмінність"
    print(f"    KS-test залишків  [{cur}]: D={d:.4f} p={p:.4f} - {verdict}")

    deg = 2 if tr["best"] == "quad" else 1
    p_synth = np.polyfit(x, y_synth, deg)
    print(f"    Коефіцієнти тренду ({tr['best']}) real: {np.round(tr['p_best'], 6)}")
    print(f"    Коефіцієнти тренду ({tr['best']}) synth: {np.round(p_synth, 6)}")
    print(f"    Автокореляція lag-1 залишків: real={lag1_autocorr(r_real):.3f}, "
          f"synth={lag1_autocorr(r_synth):.3f}")

    for key, name in [("mean", "mean"), ("std", "std"), ("skewness", "skew"), ("kurtosis", "kurt")]:
        print(f"    Різниця {name:5s}: {abs(st_real[key] - st_synth[key]):.3f}")
    print()


# plots

def _new_fig(n_cur, horizontal=False):
    if horizontal:
        return plt.subplots(1, n_cur, figsize=(18, 6), facecolor="#0F1117")
    return plt.subplots(n_cur, 1, figsize=(12, 9), facecolor="#0F1117")


def _save(fig, path, msg):
    fig.tight_layout(pad=3)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"{msg}: {path}")


def plot_currency_trends(df, trends, path):
    fig, axes = _new_fig(len(CURRENCIES))
    for ax, cur in zip(axes, CURRENCIES):
        tr = trends[cur]
        ax.plot(df["date"], df[cur], color=COLORS[cur], alpha=0.8, label="real_data")
        ax.plot(df["date"], tr["y_linear"], "--", color="#FF5252", alpha=0.9,
                label=f"linear R2={tr['r2_linear']:.3f}")
        ax.plot(df["date"], tr["y_quad"], "-.", color="#FFD700", alpha=0.9,
                label=f"quad R2={tr['r2_quad']:.3f}")
        ax_style(ax, f"{cur}/UAH — курс та тренди")
        ax.set_ylabel("UAH", color="#AAAAAA", fontsize=8)
    _save(fig, path, "Trend analysis")


def plot_residual_analysis(df, trends, path):
    fig, axes = _new_fig(len(CURRENCIES))
    for ax, cur in zip(axes, CURRENCIES):
        resid = df[cur].values - trends[cur]["y_best"]
        ax.bar(df["date"], resid, width=1.5, color=COLORS[cur], alpha=0.7)
        ax.axhline(0, color="#FFFFFF", linewidth=0.8, alpha=0.5)
        sigma = np.std(resid, ddof=1)
        ax.axhline(sigma, linestyle="--", color="#FF5252", linewidth=1, alpha=0.8, label=f"+σ ({sigma:.3f})")
        ax.axhline(-sigma, linestyle="--", color="#FF5252", linewidth=1, alpha=0.8, label=f"-σ ({-sigma:.3f})")
        ax_style(ax, f"{cur} — Залишки (Реальне − {trends[cur]['best']} тренд)")
        ax.set_ylabel("Δ UAH", color="#AAAAAA", fontsize=8)
    _save(fig, path, "Residual diagnostics")


def plot_real_vs_synthetic_series(df, synthetics, path):
    fig, axes = _new_fig(len(CURRENCIES))
    for ax, cur in zip(axes, CURRENCIES):
        ax.plot(df["date"], df[cur], color=COLORS[cur], linewidth=1.2, alpha=0.85, label="real")
        ax.plot(df["date"], synthetics[cur], color="#E040FB", linewidth=1, alpha=0.75,
                linestyle="--", label="synth")
        ax_style(ax, f"{cur} — Реальні vs Синтетичні")
        ax.set_ylabel("UAH", color="#AAAAAA", fontsize=8)
    _save(fig, path, "Real vs synthetic comparison")


def _hist_kde(ax, real, synth, color):
    n_bins = min(35, max(10, len(real) // 6))
    ax.hist(real, bins=n_bins, density=True, color=color, alpha=0.7, label="real")
    ax.hist(synth, bins=n_bins, density=True, color="#E040FB", alpha=0.5, label="synth")
    xr = np.linspace(min(real.min(), synth.min()), max(real.max(), synth.max()), 300)
    ax.plot(xr, stats.gaussian_kde(real)(xr), color="#FFFFFF", linewidth=1.5, label="kde_real")
    ax.plot(xr, stats.gaussian_kde(synth)(xr), color="#FF80AB", linewidth=1.5,
            linestyle="--", label="kde_synth")
    ax.set_ylabel("Density", color="#AAAAAA", fontsize=8)


def plot_raw_value_distributions(df, synthetics, path):
    fig, axes = _new_fig(len(CURRENCIES), horizontal=True)
    for ax, cur in zip(axes, CURRENCIES):
        _hist_kde(ax, df[cur].values, synthetics[cur], COLORS[cur])
        ax_style(ax, f"{cur} — Raw value distributions")
    _save(fig, path, "Raw value distributions")


def plot_residual_distributions(df, trends, synthetics, path):
    fig, axes = _new_fig(len(CURRENCIES), horizontal=True)
    for ax, cur in zip(axes, CURRENCIES):
        trend = trends[cur]["y_best"]
        _hist_kde(ax, df[cur].values - trend, synthetics[cur] - trend, COLORS[cur])
        ax_style(ax, f"{cur} — Residual distributions")
    _save(fig, path, "Residual distributions (detrended)")


def plot_qq_normality_check(df, trends, path):
    fig, axes = _new_fig(len(CURRENCIES), horizontal=True)
    for ax, cur in zip(axes, CURRENCIES):
        resid = df[cur].values - trends[cur]["y_best"]
        (osm, osr), (slope, intercept, r) = stats.probplot(resid, dist="norm", fit=True)
        ax.scatter(osm, osr, color=COLORS[cur], s=6, alpha=0.7, label="residuals")
        line_x = np.array([osm[0], osm[-1]])
        ax.plot(line_x, slope * line_x + intercept, color="#FF5252", linewidth=1.5, label=f"r={r:.3f}")
        ax_style(ax, f"{cur} — Q-Q plot залишків")
        ax.set_xlabel("Теоретичні квантилі", color="#AAAAAA", fontsize=8)
        ax.set_ylabel("Вибіркові квантилі", color="#AAAAAA", fontsize=8)
    _save(fig, path, "Q-Q normality checks")


def ax_style(ax, title=""):
    ax.set_facecolor("#1A1D27")
    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    if title:
        ax.set_title(title, color="#FFFFFF", fontsize=10, fontweight="bold", pad=5)
    ax.margins(x=0.01)
    ax.legend(fontsize=7, facecolor="#1A1D27", edgecolor="#444444", labelcolor="#FFFFFF", loc="upper left")
    ax.grid(linestyle="--", color="#555566", alpha=0.15, linewidth=0.6)


if __name__ == "__main__":
    main()
