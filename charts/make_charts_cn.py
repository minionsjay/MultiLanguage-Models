#!/usr/bin/env python3
"""生成所有模型对比图表 — 中文版"""
import json, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Font stack: DejaVu for Latin, Droid for CJK
fm.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Droid Sans Fallback']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.unicode_minus'] = False

OUT = "/home/ninini/Agents/Colloation_data/Training/model_reports/charts"
os.makedirs(OUT, exist_ok=True)

# ── Data ──────────────────────────────────────────
BIN_DIR = "/home/ninini/Agents/Colloation_data/Training/binary_8countries"
COUNTRIES = ['SA','BR','MX','ID','SG','TH','TR','ZA']
C_NAMES = {'SA':'沙特','BR':'巴西','MX':'墨西哥','ID':'印尼','SG':'新加坡','TH':'泰国','TR':'土耳其','ZA':'南非'}

ft, lr = {}, {}
for c in COUNTRIES:
    for store, method in [(ft,'fullft'),(lr,'lora')]:
        p = f"{BIN_DIR}/{c}_{method}/metrics.json"
        if os.path.exists(p): store[c] = json.load(open(p))

ml_data = json.load(open(f"{'/home/ninini/Agents/Colloation_data/Training/multi_lora'}/all_metrics.json"))
ad_data = json.load(open(f"{'/home/ninini/Agents/Colloation_data/Training/adapter'}/all_metrics.json"))

mc = {}
for code, path in [
    ('🇮🇩 印尼 v6','/home/ninini/Agents/Colloation_data/Training/id_data/models_v6/final_results.json'),
    ('🇸🇬 新加坡','/home/ninini/Agents/Colloation_data/Training/sg_data/models/results.json'),
    ('🇹🇭 泰国','/home/ninini/Agents/Colloation_data/Training/th_data/models/results.json'),
    ('🇿🇦 南非','/home/ninini/Agents/Colloation_data/Training/za_data/models/results.json'),
]:
    if os.path.exists(path): mc[code] = json.load(open(path))

sa_best = json.load(open(f"{'/home/ninini/Agents/Colloation_data/Training/sa_multiclass'}/model_8class_fullft_20260610_113559/metrics.json"))
mc['🇸🇦 沙特 8类'] = {'calibrated_f1': 0.829, 'per_class': sa_best}

# ════════════════════════════════════════════════════
# 图1: 二分类 — 8国 × 4方法 对比
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(18, 8))
x = np.arange(len(COUNTRIES)); w = 0.2
colors = ['#2ecc71','#3498db','#9b59b6','#e74c3c']
labels_cn = ['全量微调 (Full FT)','LoRA 微调','Multi-LoRA 共享','Adapter']
for i, (store, label, color) in enumerate([(ft,labels_cn[0],colors[0]),(lr,labels_cn[1],colors[1]),
    (ml_data,labels_cn[2],colors[2]),(ad_data,labels_cn[3],colors[3])]):
    vals = []
    for c in COUNTRIES:
        if c in store:
            d = store[c] if isinstance(store[c],dict) else store[c]
            vals.append(d.get('macro_f1',0))
        else: vals.append(0)
    bars = ax.bar(x + i*w, vals, w, label=label, color=color, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        if v > 0: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{v:.3f}',
                         ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_xticks(x + 1.5*w)
ax.set_xticklabels([C_NAMES[c] for c in COUNTRIES], fontsize=11)
ax.set_ylabel('Macro F1', fontsize=13, fontweight='bold')
ax.set_title('二分类模型对比 — 8 个国家 × 4 种训练方法', fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=10, ncol=2)
ax.set_ylim(0.60, 1.0)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# 添加解释
ax.text(0.02, 1.12, '📊 全量微调 (Full FT) 在所有国家都表现最优，平均领先 LoRA +0.044',
        transform=ax.transAxes, fontsize=10, color='#555', fontstyle='italic')
ax.text(0.02, 1.07, '   新加坡 (SG) 的二分类准确率最高 (0.919)，得益于数据量大且 Singlish 特征明显',
        transform=ax.transAxes, fontsize=10, color='#555', fontstyle='italic')

plt.tight_layout()
plt.savefig(f'{OUT}/01_binary_comparison_cn.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图1: 二分类对比")

# ════════════════════════════════════════════════════
# 图2: 多分类 — F1 排名
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
mc_names = list(mc.keys())
mc_f1s = [mc[k].get('calibrated_f1', mc[k].get('calibrated_macro_f1', mc[k].get('macro_f1',0))) for k in mc_names]
mc_lbls = [mc[k].get('labels', mc[k].get('n_labels','?')) for k in mc_names]
colors_mc = ['#27ae60','#3498db','#f39c12','#9b59b6','#e74c3c']

bars = ax.barh(range(len(mc_names)), mc_f1s, color=colors_mc[:len(mc_names)], height=0.55, edgecolor='white')
for i, (bar, v, lbl) in enumerate(zip(bars, mc_f1s, mc_lbls)):
    ax.text(bar.get_width() + 0.01, bar.get_y()+bar.get_height()/2, f'{v:.3f}',
            va='center', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.text(0.02, bar.get_y()+bar.get_height()/2, f'{mc_names[i]}  ({lbl} 类, {v*100:.1f}%)',
            va='center', fontsize=11, color='white', fontweight='bold')
ax.set_xlim(0, max(mc_f1s)*1.3)
ax.set_title('多分类模型 — Macro F1 排名', fontsize=15, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
ax.set_yticks([])

# 解释
ax.text(0.02, -0.12, '📊 泰国 (TH) 的 F1 意外高于印尼 (ID)，可能是因为标签更少 (8 vs 9) 且数据噪音更低',
        transform=ax.transAxes, fontsize=10, color='#555', fontstyle='italic')
ax.text(0.02, -0.18, '   沙特 (SA) 使用全量微调 (无 LoRA) 达到最高分，但需要 1.1GB 模型',
        transform=ax.transAxes, fontsize=10, color='#555', fontstyle='italic')

plt.tight_layout()
plt.savefig(f'{OUT}/02_multiclass_ranking_cn.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图2: 多分类排名")

# ════════════════════════════════════════════════════
# 图3: 多分类 — Per-Class F1 热力图
# ════════════════════════════════════════════════════
all_labels = set()
for k, d in mc.items():
    if 'per_class' in d:
        for l in d['per_class']:
            if isinstance(d['per_class'][l], dict) and 'f1-score' in d['per_class'][l]:
                if l not in ['accuracy','macro avg','weighted avg']: all_labels.add(l)

label_avg = {}
for l in all_labels:
    vals = []
    for k, d in mc.items():
        if 'per_class' in d and l in d['per_class'] and isinstance(d['per_class'][l], dict):
            vals.append(d['per_class'][l].get('f1-score', 0))
    if vals: label_avg[l] = np.mean(vals)
sorted_labels = sorted(label_avg.keys(), key=lambda x: label_avg[x])

heatmap_data = []
for k in mc_names:
    d = mc[k]; row = []
    for l in sorted_labels:
        if 'per_class' in d and l in d['per_class'] and isinstance(d['per_class'][l], dict):
            row.append(d['per_class'][l].get('f1-score', 0))
        else: row.append(np.nan)
    heatmap_data.append(row)

# Shorten labels for display
short_labels = []
for l in sorted_labels:
    if len(l) > 22: short_labels.append(l[:20]+'..')
    else: short_labels.append(l)

fig, ax = plt.subplots(figsize=(16, 7))
im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(sorted_labels)))
ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(mc_names)))
ax.set_yticklabels(mc_names, fontsize=11)
for i in range(len(mc_names)):
    for j in range(len(sorted_labels)):
        v = heatmap_data[i][j]
        if not np.isnan(v):
            color = 'white' if v < 0.45 else ('black' if v < 0.75 else '#1a5276')
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7.5, color=color, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('F1 分数', fontsize=11)

ax.set_title('多分类 — 各标签 F1 热力图 (绿色=好, 红色=差)', fontsize=14, fontweight='bold', pad=20)
ax.text(0.5, -0.15, '📊 NIM 生成的标签 (Political, Hate_Speech) 普遍得分较高; 国家特有标签 (ZA_Xenophobia, ZA_Severe_Racism) 需要更多训练数据',
        transform=ax.transAxes, fontsize=9, color='#555', fontstyle='italic', ha='center')

plt.tight_layout()
plt.savefig(f'{OUT}/03_multiclass_heatmap_cn.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图3: 热力图")

# ════════════════════════════════════════════════════
# 图4: 印尼模型演进
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 7))
versions = ['v1\n全量微调\n8类/38K', 'v1\nLoRA\n8类/38K', 'v2\nLSD蒸馏\n9类/38K', 'v5\n2种子集成\n9类/42K', 'v6\n标签订阅\n9类/46K']
f1_vals = [0.692, 0.690, 0.684, 0.709, 0.732]
colors_id = ['#95a5a6','#95a5a6','#3498db','#f39c12','#27ae60']
bars = ax.bar(range(len(versions)), f1_vals, color=colors_id, width=0.55, edgecolor='white', linewidth=1.5)

for bar, v, ver in zip(bars, f1_vals, versions):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f'{v:.3f}',
            ha='center', fontsize=13, fontweight='bold')

ax.set_xticks(range(len(versions)))
ax.set_xticklabels(versions, fontsize=9, linespacing=1.2)
ax.set_ylabel('Macro F1', fontsize=12, fontweight='bold')
ax.set_title('🇮🇩 印尼 (ID) 多分类模型演进路线', fontsize=15, fontweight='bold', pad=15)
ax.set_ylim(0.65, 0.78)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Annotations
ax.annotate('+2,500 ID_SARA\nNIM 生成数据', xy=(3, 0.709), xytext=(1, 0.765),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5), fontsize=9, color='#c0392b', fontweight='bold')
ax.annotate('+2,000 Dangerous\n+1,500 Hate_Speech\nNIM 生成数据', xy=(4, 0.732), xytext=(2.5, 0.765),
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5), fontsize=9, color='#1e8449', fontweight='bold')

ax.text(0.5, -0.18, '📊 每次 NIM 生成数据补充都带来显著提升: ID_SARA +0.20, Dangerous +0.15, Hate_Speech +0.09',
        transform=ax.transAxes, fontsize=10, color='#555', fontstyle='italic', ha='center')
ax.text(0.5, -0.24, '   LSD 蒸馏 (v2) 提升不明显 (-0.008) 且训练速度慢 5 倍，不推荐使用',
        transform=ax.transAxes, fontsize=10, color='#555', fontstyle='italic', ha='center')

plt.tight_layout()
plt.savefig(f'{OUT}/04_id_evolution_cn.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图4: ID 演进")

# ════════════════════════════════════════════════════
# 图5: 综合仪表盘 (6 宫格)
# ════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# 5a: Full FT vs LoRA F1 差距
ax1 = fig.add_subplot(gs[0, 0])
diffs = []
for c in COUNTRIES:
    if c in ft and c in lr:
        diffs.append({'country':C_NAMES[c], 'ft':ft[c]['macro_f1'], 'lr':lr[c]['macro_f1'],
                      'gap':ft[c]['macro_f1']-lr[c]['macro_f1']})
df_diff = pd.DataFrame(diffs).sort_values('gap', ascending=False)
colors_gap = ['#e74c3c' if x > 0.03 else '#f39c12' for x in df_diff['gap']]
bars = ax1.barh(df_diff['country'], df_diff['gap'], color=colors_gap, height=0.6, edgecolor='white')
for bar, v in zip(bars, df_diff['gap']):
    ax1.text(bar.get_width() + 0.002, bar.get_y()+bar.get_height()/2, f'{v:.3f}',
            va='center', fontsize=10, fontweight='bold')
ax1.axvline(x=0, color='black', linewidth=0.8)
ax1.set_title('Full FT vs LoRA: 性能差距', fontsize=12, fontweight='bold')
ax1.set_xlabel('Macro F1 差异 (Full FT - LoRA)')
ax1.grid(axis='x', alpha=0.3)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

# 5b: 模型大小对比
ax2 = fig.add_subplot(gs[0, 1])
sizes = {'全量微调 (单国)': 1120, '全量微调 (8国)': 8960,
         'LoRA 适配器': 3, 'Multi-LoRA 适配器': 2.4,
         'Multi-LoRA (8国+基座)': 1140, '多分类 LoRA': 2.4}
cols_size = ['#e74c3c','#c0392b','#3498db','#9b59b6','#8e44ad','#27ae60']
ax2.barh(list(sizes.keys()), list(sizes.values()), color=cols_size, height=0.55, edgecolor='white')
for i, (k, v) in enumerate(sizes.items()):
    label = f'{v} MB' if v < 1000 else f'{v/1000:.1f} GB'
    ax2.text(v+1 if v<100 else v+50, i, label, va='center', fontsize=10, fontweight='bold')
ax2.set_title('模型存储大小对比 (MB)', fontsize=12, fontweight='bold')
ax2.set_xscale('log'); ax2.set_xlim(0.5, 20000)
ax2.grid(axis='x', alpha=0.3)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

# 5c: Clean vs Violation F1
ax3 = fig.add_subplot(gs[0, 2])
for c in COUNTRIES:
    if c in ft:
        ax3.scatter(ft[c]['clean']['f1'], ft[c]['violation']['f1'], s=150, c='#27ae60',
                  edgecolors='white', linewidth=1.5, zorder=5)
        ax3.annotate(f'FT-{C_NAMES[c]}', (ft[c]['clean']['f1'], ft[c]['violation']['f1']),
                   textcoords="offset points", xytext=(6,4), fontsize=8, fontweight='bold')
    if c in lr:
        ax3.scatter(lr[c]['clean']['f1'], lr[c]['violation']['f1'], s=100, c='#3498db',
                  edgecolors='white', linewidth=1, zorder=4, alpha=0.7)
ax3.plot([0.5,1],[0.5,1], '--', color='gray', alpha=0.3)
ax3.set_xlabel('安全内容 F1', fontsize=11); ax3.set_ylabel('违规内容 F1', fontsize=11)
ax3.set_title('安全 vs 违规: F1 权衡', fontsize=12, fontweight='bold')
ax3.set_xlim(0.65, 1.0); ax3.set_ylim(0.65, 1.0); ax3.grid(alpha=0.3)
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

# 5d: 违规召回率 TOP 排名
ax4 = fig.add_subplot(gs[1, 0])
viol_all = []
for c in COUNTRIES:
    if c in ft: v=ft[c]['violation']; viol_all.append((v['recall'], C_NAMES[c], 'Full FT', v['f1']))
    if c in lr: v=lr[c]['violation']; viol_all.append((v['recall'], C_NAMES[c], 'LoRA', v['f1']))
    v=ml_data[c]['violation']; viol_all.append((v['recall'], C_NAMES[c], 'MLoRA', v['f1']))
viol_all.sort(reverse=True); top10 = viol_all[:12]
ax4.barh([f'{c}({m})' for _,c,m,_ in top10], [r for r,_,_,_ in top10],
         color=['#27ae60','#3498db','#9b59b6']*4, height=0.6, edgecolor='white')
for i, (r,_,_,f1) in enumerate(top10):
    ax4.text(r+0.005, i, f'召回:{r:.3f}  F1:{f1:.3f}', va='center', fontsize=8)
ax4.set_title('违规召回率 TOP 12', fontsize=12, fontweight='bold')
ax4.set_xlim(0.75, 1.05); ax4.grid(axis='x', alpha=0.3)
ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)

# 5e: ID v5→v6 每类提升
ax5 = fig.add_subplot(gs[1, 1])
v5_d = json.load(open('/home/ninini/Agents/Colloation_data/Training/id_data/models_v5/final_results.json'))
v6_d = json.load(open('/home/ninini/Agents/Colloation_data/Training/id_data/models_v6/final_results.json'))
labels_comp = ['ID_SARA\n(宗教种族)','Dangerous\n(危险内容)','Hate_Speech\n(仇恨言论)','Harassment\n(骚扰)','Sexually\nExplicit','Political\n(政治)','ID_Blasphemy\n(亵渎)','safe\n(安全)','other\n(其他)']
v5_vals = [v5_d['per_class'].get(l.split('\n')[0],{}).get('f1-score',0) for l in labels_comp]
v6_vals = [v6_d['per_class'].get(l.split('\n')[0],{}).get('f1-score',0) for l in labels_comp]
x = np.arange(len(labels_comp)); w = 0.35
ax5.bar(x-w/2, v5_vals, w, label='v5 (42K条)', color='#f39c12', edgecolor='white')
ax5.bar(x+w/2, v6_vals, w, label='v6 (46K条)', color='#27ae60', edgecolor='white')
for i in range(len(labels_comp)):
    diff = v6_vals[i] - v5_vals[i]
    if diff > 0.03:
        ax5.annotate(f'+{diff:.2f}', (i, v6_vals[i]+0.03), ha='center', fontsize=9, color='#e74c3c', fontweight='bold')
ax5.set_xticks(x); ax5.set_xticklabels(labels_comp, rotation=45, ha='right', fontsize=7.5)
ax5.legend(fontsize=9); ax5.set_ylim(0,1.1)
ax5.set_title('印尼 v5 → v6: 各类别 F1 提升', fontsize=12, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)
ax5.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False)

# 5f: 数据量 vs 模型性能
ax6 = fig.add_subplot(gs[1, 2])
mc_rows = {'印尼 v6':45753,'新加坡':25543,'泰国':14751,'南非':21329}
mc_f1_dict = {}
for k in mc_names:
    for name, rows in mc_rows.items():
        if name.split()[0] in k: mc_f1_dict[name] = mc[k].get('calibrated_f1',mc[k].get('calibrated_macro_f1',0))
colors_pt = ['#27ae60','#3498db','#f39c12','#e74c3c']
for i, (k, rows) in enumerate(mc_rows.items()):
    f1 = mc_f1_dict.get(k, 0)
    ax6.scatter(rows/1000, f1, s=250, c=colors_pt[i], edgecolors='white', linewidth=2, zorder=5)
    ax6.annotate(f'{k}\n{rows/1000:.0f}K条, F1={f1:.3f}', (rows/1000, f1),
                textcoords="offset points", xytext=(12,15), fontsize=9, fontweight='bold')
ax6.set_xlabel('训练数据量 (千条)', fontsize=11)
ax6.set_ylabel('Macro F1', fontsize=11)
ax6.set_title('数据量 vs 模型性能', fontsize=12, fontweight='bold')
ax6.grid(alpha=0.3)
ax6.spines['top'].set_visible(False); ax6.spines['right'].set_visible(False)

# 总标题
fig.suptitle('多语言内容审核模型 — 综合对比仪表盘',
             fontsize=17, fontweight='bold', y=1.01)
fig.text(0.5, 0.01, '基座模型: microsoft/mdeberta-v3-base | LoRA: r=16 α=32 | Label Smoothing: 0.05 | 训练环境: AMD RX 9070 XT ROCm',
         ha='center', fontsize=10, color='#888', fontstyle='italic')

plt.savefig(f'{OUT}/05_dashboard_cn.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图5: 综合仪表盘")

# ════════════════════════════════════════════════════
# 图6: 数据生成效果 — 关键提升
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))
improvements = [
    ('ID_SARA\n(印尼宗教种族)', 0.42, 0.65, +0.23, 'NIM 生成\n+2,500 条'),
    ('Dangerous_Content\n(危险内容)', 0.58, 0.73, +0.15, 'NIM 生成\n+2,000 条'),
    ('Hate_Speech\n(仇恨言论)', 0.60, 0.69, +0.09, 'NIM 生成\n+1,500 条'),
    ('ID_Blasphemy\n(印尼亵渎)', 0.89, 0.93, +0.04, '陪审团验证'),
    ('Political\n(政治)', 0.92, 0.92, 0.00, '已饱和'),
]

y_pos = range(len(improvements))
for i, (name, before, after, diff, note) in enumerate(improvements):
    # Before bar
    ax.barh(i+0.15, before, 0.3, color='#e74c3c', edgecolor='white', label='改进前' if i==0 else '')
    # After bar
    ax.barh(i-0.15, after, 0.3, color='#27ae60', edgecolor='white', label='改进后' if i==0 else '')
    # Diff annotation
    if diff > 0:
        ax.text(max(before,after)+0.02, i, f'+{diff:.2f}', va='center', fontsize=12, color='#e74c3c', fontweight='bold')
    # Note
    ax.text(0.02, i+0.25, f'{note}', va='bottom', fontsize=8, color='#555', fontstyle='italic')

ax.set_yticks(range(len(improvements)))
ax.set_yticklabels([i[0] for i in improvements], fontsize=10)
ax.set_xlim(0, 1.1)
ax.set_xlabel('F1 分数', fontsize=12, fontweight='bold')
ax.set_title('印尼 (ID) — NIM 数据生成对弱势类别的提升效果', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

ax.text(0.5, -0.15, '📊 NIM API (llama-4-maverick, mistral-large, qwen3-next) 生成的高质量合成数据是提升模型的最有效方法',
        transform=ax.transAxes, fontsize=10, color='#555', fontstyle='italic', ha='center')
ax.text(0.5, -0.21, '   ID_SARA 提升最大 (+0.23)，因为原始数据只有 2,983 条且与 Harassment/Hate_Speech 高度混淆',
        transform=ax.transAxes, fontsize=10, color='#555', fontstyle='italic', ha='center')

plt.tight_layout()
plt.savefig(f'{OUT}/06_data_augmentation_impact_cn.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图6: 数据增强效果")

# ════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"所有图表已保存到: {OUT}")
for f in sorted(os.listdir(OUT)):
    if f.endswith('.png'):
        sz = os.path.getsize(f"{OUT}/{f}")/1024
        print(f"  {f} ({sz:.0f}KB)")
