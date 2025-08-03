import matplotlib.pyplot as plt
import numpy as np

# Data
clusters = ["Baseline", "2", "3", "4", "5"]
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

# Loop untuk membuat satu gambar untuk setiap domain
for dom in domains:
    # 1. Membuat figur baru untuk setiap iterasi
    fig, ax = plt.subplots(figsize=(11, 6))
    
    y = np.arange(len(clusters))
    
    # Mengambil data khusus untuk domain ini
    domain_f1_src = f1_src[dom]
    domain_f1_tgt = f1_tgt[dom]

    # Plot garis dumbbell
    ax.hlines(y, domain_f1_src, domain_f1_tgt, color="gray", lw=3, alpha=0.6)
    
    # Plot titik data
    p1 = ax.scatter(domain_f1_src, y, marker="o", s=150, color=clr_src, zorder=3, label="F1-Source")
    p2 = ax.scatter(domain_f1_tgt, y, marker="s", s=150, color=clr_tgt, zorder=3, label="F1-Target")

    # Anotasi persentase perubahan
    for j, (src_val, tgt_val) in enumerate(zip(domain_f1_src, domain_f1_tgt)):
        gap = (tgt_val - src_val) / src_val * 100
        offset = 0.02
        text_pos = tgt_val + offset if gap > 0 else tgt_val - offset
        ha = 'left' if gap > 0 else 'right'
        
        if dom == "CSE-CIC" and clusters[j] in ["4"]:
            text_pos = tgt_val - offset
            ha = 'right'
        ax.text(text_pos, y[j], f"{gap:+.1f}%", va="center", ha=ha, fontsize=12)

    # 2. Menambahkan judul spesifik untuk setiap plot
    ax.set_title(f"Analisis F1-Score untuk Target Domain: {dom}", fontsize=18, pad=20)
    
    # Pengaturan subplot
    ax.set_yticks(y)
    ax.set_yticklabels(clusters)
    ax.set_xlabel("F1-Score", fontsize=14)
    ax.set_ylabel("Klaster", fontsize=14)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.spines[['right', 'top']].set_visible(False)
    
    # Menambahkan garis referensi
    
    # 3. Menambahkan legenda yang simpel dan andal
    ax.legend(handles=[p1,p2], loc='upper center', bbox_to_anchor=(0.5, -0.15), 
           ncol=4, frameon=False, fontsize=12)
    
    plt.xticks(np.arange(0.0, 1.1, 0.1))
    plt.tight_layout()
    
    # 4. Menyimpan setiap gambar dengan nama file yang unik
    # Mengganti '-' dengan '_' agar nama file valid
    safe_dom_name = dom.replace('‑', '_')
    plt.savefig(f"diagram_terpisah_{safe_dom_name}.png", dpi=300)
    
    plt.show()
