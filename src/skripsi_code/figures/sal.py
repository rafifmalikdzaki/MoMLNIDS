import matplotlib.pyplot as plt
import numpy as np

# Data
domains = ["ToN‑IoT", "CSE‑CIC", "UNSW‑NB15"]
clusters = np.array([2, 3, 4, 5])

baseline_src = np.array([0.9733, 0.7140, 0.6595])
f1_src_values = {
    "ToN‑IoT": np.array([0.9580, 0.9562, 0.8345, 0.8372]),
    "CSE‑CIC": np.array([0.6748, 0.5472, 0.6796, 0.7426]),
    "UNSW‑NB15": np.array([0.6949, 0.6905, 0.7302, 0.7009]),
}
delta_src_pct = {
    "ToN‑IoT": (f1_src_values["ToN‑IoT"] - baseline_src[0]) / baseline_src[0] * 100,
    "CSE‑CIC": (f1_src_values["CSE‑CIC"] - baseline_src[1]) / baseline_src[1] * 100,
    "UNSW‑NB15": (f1_src_values["UNSW‑NB15"] - baseline_src[2]) / baseline_src[2] * 100,
}

# Palet warna dan gaya
colors = {"ToN‑IoT": "#E69F00", "CSE‑CIC": "#56B4E9", "UNSW‑NB15": "#009E73"}
plt.style.use('seaborn-v0_8-whitegrid')

# --- Figure 2: Line Chart dengan legenda di bawah ---
fig2, ax2 = plt.subplots(figsize=(10, 7)) # Menambah tinggi figur untuk ruang legenda

for dom in domains:
    ax2.plot(clusters, delta_src_pct[dom], marker='o', lw=2.5, color=colors[dom], label=dom, markersize=8)

ax2.axhline(0, linestyle='--', color='black', lw=1.5, alpha=0.7)
ax2.set_xlabel("Jumlah Klaster Pseudo‑Domain", fontsize=12)
ax2.set_ylabel("Perubahan F1-Source (%)", fontsize=12)
ax2.set_title("Perubahan Relatif F1-Score Source dari Baseline", fontsize=16)
ax2.set_xticks(clusters)

# Mengatur legenda di bawah sumbu-X, secara horizontal
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
           ncol=3, frameon=False, fontsize=12)

# Menyesuaikan layout agar legenda tidak terpotong
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
