import matplotlib.pyplot as plt
import numpy as np

# Data
domains = ["ToN‑IoT", "CSE‑CIC", "UNSW‑NB15"]
clusters = np.array([2, 3, 4, 5])
delta_pct = {
    "ToN‑IoT": np.array([-22.2, 6.5, -23.3, -23.3]),
    "CSE‑CIC": np.array([-21.8, 85.7, 92.6, 33.7]),
    "UNSW‑NB15": np.array([-2.2, -9.0, -8.4, 4.7]),
}

# Palet warna dan gaya
colors = {"ToN‑IoT": "#E69F00", "CSE‑CIC": "#56B4E9", "UNSW‑NB15": "#009E73"}
plt.style.use('seaborn-v0_8-whitegrid')

# --- Figure 2: Line Chart dengan legenda di bawah ---
fig2, ax2 = plt.subplots(figsize=(10, 7))

for dom in domains:
    ax2.plot(clusters, delta_pct[dom], marker='o', lw=2.5, color=colors[dom], label=dom, markersize=8)

ax2.axhline(0, linestyle='--', color='black', lw=1.5, alpha=0.7)
ax2.set_xlabel("Jumlah Klaster Pseudo‑Domain", fontsize=12)
ax2.set_ylabel("Perubahan F1-Target (%)", fontsize=12)
ax2.set_title("Perubahan Relatif F1-Score Target dari Baseline", fontsize=16)
ax2.set_xticks(clusters)

# Mengatur legenda di bawah sumbu-X, secara horizontal
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
           ncol=3, frameon=False, fontsize=12)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
