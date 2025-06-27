import matplotlib.pyplot as plt
import numpy as np

# Data
clusters = ["Base", "2", "3", "4", "5"]
domains = ["ToN-IoT", "CSE-CIC", "UNSW-NB15"]

f1_src = {
    "ToN-IoT": [0.9733, 0.9580, 0.9562, 0.8345, 0.8372],
    "CSE-CIC": [0.7140, 0.6748, 0.5472, 0.6796, 0.7426],
    "UNSW-NB15": [0.6595, 0.6949, 0.6905, 0.7302, 0.7009],
}
f1_tgt = {
    "ToN-IoT": [0.2469, 0.1920, 0.2630, 0.1894, 0.1894],
    "CSE-CIC": [0.3529, 0.2759, 0.6555, 0.6798, 0.4717],
    "UNSW-NB15": [0.8986, 0.8784, 0.8176, 0.8229, 0.9409],
}

# Palet warna dan gaya
plt.style.use('seaborn-v0_8-talk')
clr_src, clr_tgt = "#0072B2", "#D55E00"

# Membuat subplot
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 15), sharex=True)

for i, dom in enumerate(domains):
    ax = axes[i]
    y = np.arange(len(clusters))

    # Plot garis dumbbell
    ax.hlines(y, f1_src[dom], f1_tgt[dom], color="gray", lw=3, alpha=0.6)
    
    # Plot titik data
    ax.scatter(f1_src[dom], y, marker="o", s=150, color=clr_src, zorder=3, label="Source")
    ax.scatter(f1_tgt[dom], y, marker="s", s=150, color=clr_tgt, zorder=3, label="Target")

    # Anotasi persentase perubahan dengan posisi yang disesuaikan
    for j, (src_val, tgt_val) in enumerate(zip(f1_src[dom], f1_tgt[dom])):
        gap = (tgt_val - src_val) / src_val * 100
        # Menambah jarak horizontal agar tidak menabrak titik
        offset = 0.015 
        text_pos = tgt_val + offset if gap > 0 else tgt_val - offset
        ha = 'left' if gap > 0 else 'right'
        
        # Penyesuaian khusus untuk kasus yang sangat berdekatan di domain CSE-CIC
        if dom == "CSE-CIC" and clusters[j] in ["4"]:
             text_pos = tgt_val - offset
             ha = 'right'

        ax.text(text_pos, y[j], f"{gap:+.1f}%", va="center", ha=ha, fontsize=12)

    # Pengaturan subplot
    ax.set_yticks(y)
    ax.set_yticklabels(clusters)
    ax.set_title(f"Domain Target: {dom}", fontsize=16)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.spines[['right', 'top']].set_visible(False)

    
    # Menambahkan garis referensi
    ax.axvline(0.5, color='red', linestyle='--', lw=1, alpha=0.5)
    ax.axvline(1.0, color='green', linestyle='--', lw=1, alpha=0.5)

# 1. Atur Judul Utama. Posisi y default (sekitar 0.98) sudah cukup baik.
fig.suptitle("Perbandingan F1-Score Source vs Target di Berbagai Domain", fontsize=20)

# 2. Atur legenda agar berada di bawah judul.
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, 
           bbox_to_anchor=(0.5, 0.96)) # Turunkan posisi legenda sedikit

# 3. Biarkan tight_layout bekerja secara otomatis tanpa parameter rect
plt.tight_layout()

# 4. Beri sedikit ruang tambahan secara manual jika perlu setelah tight_layout
fig.subplots_adjust(top=0.88) # Beri ruang di atas untuk judul dan legenda

plt.xlabel("F1-Score", fontsize=14)
plt.xticks(np.arange(0, 1.1, 0.1), fontsize=12)
# plt.xlim(0, 1.0) # xlim bisa diatur setelahnya jika perlu

plt.savefig("enhanced_visualization_fixed_labels.png", dpi=300, bbox_inches='tight')
plt.show()
