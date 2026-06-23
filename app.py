import streamlit as st
import joblib
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

# ── Load assets ───────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model       = joblib.load("assets/xgb_model.pkl")
    features    = joblib.load("assets/features_v2.pkl")
    team_stats  = joblib.load("assets/team_stats.pkl")
    match_proba = joblib.load("assets/match_proba.pkl")
    groups      = joblib.load("assets/groups.pkl")
    wc2026_teams= joblib.load("assets/wc2026_teams.pkl")
    mc_results  = pd.read_csv("assets/monte_carlo_results.csv")
    return model, features, team_stats, match_proba, groups, wc2026_teams, mc_results

model, features, team_stats, match_proba, groups, wc2026_teams, mc_results = load_assets()

# ── Fungsi simulasi ───────────────────────────────────────────
def simulate_match_fast(home, away):
    proba = match_proba[(home, away)]
    return random.choices(["home_win", "draw", "away_win"], weights=proba)[0]

def simulate_group_stage(groups):
    top2, third_place = [], []
    for _, teams_in_group in groups.items():
        points = {t: 0 for t in teams_in_group}
        gd     = {t: 0 for t in teams_in_group}
        for i in range(len(teams_in_group)):
            for j in range(i + 1, len(teams_in_group)):
                home, away = teams_in_group[i], teams_in_group[j]
                outcome = simulate_match_fast(home, away)
                if outcome == "home_win":
                    points[home] += 3; gd[home] += 1; gd[away] -= 1
                elif outcome == "away_win":
                    points[away] += 3; gd[away] += 1; gd[home] -= 1
                else:
                    points[home] += 1; points[away] += 1
        standings = sorted(teams_in_group, key=lambda t: (points[t], gd[t]), reverse=True)
        top2.extend(standings[:2])
        third_place.append((standings[2], points[standings[2]], gd[standings[2]]))
    best_third = [t[0] for t in sorted(third_place, key=lambda x: (x[1], x[2]), reverse=True)[:8]]
    qualifiers = top2 + best_third
    random.shuffle(qualifiers)
    return qualifiers

def simulate_knockout(teams_list, track=False):
    current_round = teams_list[:]
    finalist, semifinalist = [], []
    round_num = 0
    while len(current_round) > 1:
        next_round = []
        if len(current_round) == 4:
            semifinalist = current_round[:]
        if len(current_round) == 2:
            finalist = current_round[:]
        for i in range(0, len(current_round), 2):
            home, away = current_round[i], current_round[i+1]
            while True:
                outcome = simulate_match_fast(home, away)
                if outcome != "draw": break
            next_round.append(home if outcome == "home_win" else away)
        current_round = next_round
    return current_round[0], finalist, semifinalist

def run_monte_carlo(n=3000):
    champion_c  = Counter()
    finalist_c  = Counter()
    semifinal_c = Counter()
    for _ in range(n):
        qualifiers = simulate_group_stage(groups)
        champ, finalists, semis = simulate_knockout(qualifiers, track=True)
        champion_c[champ] += 1
        for t in finalists:  finalist_c[t]  += 1
        for t in semis:      semifinal_c[t] += 1
    return champion_c, finalist_c, semifinal_c

# ── UI ────────────────────────────────────────────────────────
st.set_page_config(page_title="WC 2026 Predictor", page_icon="⚽", layout="wide")

page = st.sidebar.radio(
    "Navigasi",
    ["🏠 Home", "📊 Team Analytics", "⚔️ Match Predictor", "🏆 WC Simulator"]
)

# ── Halaman Home ──────────────────────────────────────────────
if page == "🏠 Home":
    st.title("⚽ FIFA World Cup 2026 Predictor")
    st.markdown("""
    Dashboard prediksi Piala Dunia FIFA 2026 menggunakan **Machine Learning** dan **Monte Carlo Simulation**.
    
    ### Fitur:
    - 📊 **Team Analytics** — statistik dan kekuatan tiap tim
    - ⚔️ **Match Predictor** — prediksi probabilitas hasil pertandingan
    - 🏆 **WC Simulator** — simulasi turnamen 10.000x
    
    ### Metodologi:
    - Model: **XGBoost** (dilatih dari 2.269 pertandingan historis, 2007–2026)
    - Simulasi: **Monte Carlo** 3.000 iterasi
    - Fitur: Win rate, avg goals, FIFA ranking, market value, form streak, H2H
    """)

    st.divider()
    st.subheader("📈 Prediksi Juara (Pre-computed 10.000x)")
    top10 = mc_results.head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top10["team"][::-1], top10["win_probability"][::-1],
            color="steelblue", edgecolor="white")
    for i, (_, row) in enumerate(top10[::-1].iterrows()):
        ax.text(row["win_probability"] + 0.001, i, f"{row['win_probability']:.1%}",
                va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Peluang Juara")
    ax.set_title("Top 10 Kandidat Juara WC 2026")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)

# ── Halaman Team Analytics ────────────────────────────────────
elif page == "📊 Team Analytics":
    st.title("📊 Team Analytics")
    selected = st.selectbox("Pilih Tim", sorted(wc2026_teams))
    stats = team_stats[selected]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏅 FIFA Ranking",   f"#{int(stats['fifa_ranking'])}")
    col2.metric("⚽ Avg Goals",       f"{stats['avg_goals']:.2f}")
    col3.metric("📈 Win Rate",        f"{stats['win_rate']:.0%}")
    col4.metric("💰 Market Value",    f"€{stats['total_market_value']/1e6:.0f}M")

    st.divider()

    # Radar chart perbandingan vs rata-rata
    st.subheader("Perbandingan vs Rata-rata WC 2026")
    all_stats = pd.DataFrame(team_stats).T
    avg = all_stats.mean()

    categories  = ["Win Rate", "Avg Goals", "Market Value (€B)", "Form Streak"]
    team_values = [
        stats["win_rate"],
        stats["avg_goals"],
        stats["total_market_value"] / 1e9,
        stats["form_streak"],
    ]
    avg_values = [
        avg["win_rate"],
        avg["avg_goals"],
        avg["total_market_value"] / 1e9,
        avg["form_streak"],
    ]

    x = np.arange(len(categories))
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.bar(x - 0.2, team_values, 0.4, label=selected,     color="steelblue")
    ax2.bar(x + 0.2, avg_values,  0.4, label="Avg WC 2026", color="lightgray")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title(f"{selected} vs Rata-rata Tim WC 2026")
    plt.tight_layout()
    st.pyplot(fig2)

    # MC result tim ini
    st.divider()
    st.subheader("Prediksi Monte Carlo")
    team_mc = mc_results[mc_results["team"] == selected]
    if not team_mc.empty:
        prob = team_mc["win_probability"].values[0]
        rank = mc_results[mc_results["team"] == selected].index[0] + 1
        st.metric("🏆 Peluang Juara", f"{prob:.2%}", f"Rank #{rank} dari 48 tim")
    else:
        st.info("Tim ini tidak muncul dalam hasil simulasi.")

# ── Halaman Match Predictor ───────────────────────────────────
elif page == "⚔️ Match Predictor":
    st.title("⚔️ Match Predictor")
    col1, col2 = st.columns(2)
    home_team = col1.selectbox("🏠 Home Team", sorted(wc2026_teams), index=0)
    away_team = col2.selectbox("✈️ Away Team", sorted(wc2026_teams), index=1)

    if home_team == away_team:
        st.warning("Pilih tim yang berbeda!")
    else:
        proba = match_proba[(home_team, away_team)]
        home_win, draw, away_win = proba[0], proba[1], proba[2]

        st.divider()
        st.subheader(f"{home_team} vs {away_team}")

        col1, col2, col3 = st.columns(3)
        col1.metric(f"🏠 {home_team} Win", f"{home_win:.1%}")
        col2.metric("🤝 Draw",             f"{draw:.1%}")
        col3.metric(f"✈️ {away_team} Win", f"{away_win:.1%}")

        # Bar chart probabilitas
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        labels = [f"{home_team} Win", "Draw", f"{away_team} Win"]
        values = [home_win, draw, away_win]
        colors = ["steelblue", "gray", "tomato"]
        bars   = ax3.bar(labels, values, color=colors, edgecolor="white", width=0.5)
        for bar, val in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.1%}", ha="center", fontweight="bold")
        ax3.set_ylim(0, max(values) * 1.3)
        ax3.set_ylabel("Probabilitas")
        ax3.spines[["top", "right"]].set_visible(False)
        ax3.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig3)

        st.divider()
        st.subheader("Perbandingan Kekuatan Tim")
        h, a = team_stats[home_team], team_stats[away_team]
        compare_df = pd.DataFrame({
            "Statistik":    ["FIFA Ranking", "Win Rate", "Avg Goals", "Market Value (€M)", "Form Streak"],
            home_team:      [int(h["fifa_ranking"]), f"{h['win_rate']:.0%}", f"{h['avg_goals']:.2f}",
                             f"€{h['total_market_value']/1e6:.0f}M", int(h["form_streak"])],
            away_team:      [int(a["fifa_ranking"]), f"{a['win_rate']:.0%}", f"{a['avg_goals']:.2f}",
                             f"€{a['total_market_value']/1e6:.0f}M", int(a["form_streak"])],
        })
        st.dataframe(compare_df, hide_index=True, use_container_width=True)

# ── Halaman WC Simulator ──────────────────────────────────────
elif page == "🏆 WC Simulator":
    st.title("🏆 World Cup 2026 Simulator")
    st.info("Simulasi menggunakan Monte Carlo — setiap klik menghasilkan hasil berbeda.")

    n_sim = st.slider("Jumlah Simulasi", min_value=500, max_value=3000, value=1000, step=500)

    if st.button("▶️ Run Simulation", type="primary"):
        with st.spinner(f"Menjalankan {n_sim} simulasi..."):
            champion_c, finalist_c, semifinal_c = run_monte_carlo(n_sim)

        st.success("Simulasi selesai!")
        st.divider()

        # Gabung hasil
        all_teams = set(list(champion_c.keys()) + list(finalist_c.keys()) + list(semifinal_c.keys()))
        result_df = pd.DataFrame([{
            "Tim":          t,
            "🏆 Juara":     f"{champion_c[t]/n_sim:.1%}",
            "🥈 Final":     f"{finalist_c[t]/n_sim:.1%}",
            "🥉 Semi Final":f"{semifinal_c[t]/n_sim:.1%}",
            "_champ_val":   champion_c[t]/n_sim
        } for t in all_teams]).sort_values("_champ_val", ascending=False).drop(columns=["_champ_val"])

        st.subheader("📋 Tabel Peluang")
        st.dataframe(result_df.reset_index(drop=True), hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("📊 Top 15 Peluang Juara")
        top15 = sorted(champion_c.items(), key=lambda x: x[1], reverse=True)[:15]
        teams_top, counts_top = zip(*top15)
        probs_top = [c/n_sim for c in counts_top]

        fig4, ax4 = plt.subplots(figsize=(9, 6))
        bars = ax4.barh(list(teams_top)[::-1], probs_top[::-1], color="steelblue", edgecolor="white")
        for bar, val in zip(bars, probs_top[::-1]):
            ax4.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                     f"{val:.1%}", va="center", fontsize=9, fontweight="bold")
        ax4.set_xlabel("Peluang Juara")
        ax4.set_title(f"Prediksi Juara WC 2026 ({n_sim} Simulasi)")
        ax4.spines[["top", "right"]].set_visible(False)
        ax4.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig4)