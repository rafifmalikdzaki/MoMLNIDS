import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Data untuk F1‑Source
domains = ["ToN‑IoT", "CSE‑CIC", "UNSW‑NB15"]
clusters = np.array([2, 3, 4, 5])

baseline_src = np.array([0.9733, 0.7140, 0.6595])
f1_src_values = {
    "ToN‑IoT": np.array([0.9580, 0.9562, 0.8345, 0.8372]),
    "CSE‑CIC": np.array([0.6748, 0.5472, 0.6796, 0.7426]),
    "UNSW‑NB15": np.array([0.6949, 0.6905, 0.7302, 0.7009]),
}

# Palet warna dan gaya
colors = {"ToN‑IoT": "#E69F00", "CSE‑CIC": "#56B4E9", "UNSW‑NB15": "#009E73"}
plt.style.use('seaborn-v0_8-whitegrid')

# --- Figure 1: Bar Chart dengan legenda di bawah ---
fig1, ax1 = plt.subplots(figsize=(12, 8)) # Menambah tinggi figur untuk ruang legenda
bar_w = 0.25
x_pos = np.arange(len(clusters))

for idx, dom in enumerate(domains):
    bars = ax1.bar(x_pos + idx * bar_w - bar_w, f1_src_values[dom], width=bar_w, color=colors[dom])
    ax1.axhline(y=baseline_src[idx], color=colors[dom], linestyle='--', lw=2, alpha=0.8)
    ax1.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)

ax1.set_xlabel("Jumlah Klaster Pseudo‑Domain", fontsize=12)
ax1.set_xticks(x_pos + bar_w / 2)
ax1.set_xticklabels(clusters)
ax1.set_ylabel("F1‑Score Source", fontsize=12)
ax1.set_title("Perbandingan F1-Score Source (Bar) vs. Baseline (Garis Putus-putus)", fontsize=16)
ax1.set_ylim(0, 1.1)
ax1.grid(axis='x', visible=False)

# Membuat legenda kustom
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, color=colors[dom], label=f'{dom} F1-Source') for dom in domains
]
legend_elements.append(
    Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Baseline')
)

# Mengatur legenda di bawah sumbu-X, secara horizontal
ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
           ncol=4, frameon=False, fontsize=12)

# Menyesuaikan layout agar legenda tidak terpotong
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
