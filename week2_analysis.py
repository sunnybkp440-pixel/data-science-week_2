import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
sns.set_theme(style="whitegrid", font_scale=1.0)
pal = {"Malignant":"#d62728","Benign":"#1f77b4"}

# ---- Load + clean (same pipeline as Week 1) ----
d = load_breast_cancer(as_frame=True).frame
d["diagnosis"] = d["target"].map({0:"Malignant",1:"Benign"}); d = d.drop(columns="target")
d.columns = [c.replace(" ","_") for c in d.columns]
num = d.select_dtypes("number").columns
q1,q3 = d[num].quantile(.25), d[num].quantile(.75); iqr=q3-q1
df = d.copy(); df[num] = df[num].clip(q1-1.5*iqr, q3+1.5*iqr, axis=1)
df["y"] = (df.diagnosis=="Malignant").astype(int)

# ---- Fig 1: violin small multiples ----
feats = ["worst_concave_points","worst_perimeter","worst_radius","mean_concavity"]
fig, ax = plt.subplots(1,4, figsize=(13,4))
for a,f in zip(ax,feats):
    sns.violinplot(data=df, x="diagnosis", y=f, hue="diagnosis", palette=pal, inner="quartile", ax=a, legend=False, order=["Benign","Malignant"])
    a.set_title(f.replace("_"," "), fontsize=10); a.set_xlabel("")
fig.suptitle("Malignant tumours sit higher on every key feature", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig("w2figs/v1_violin.png", dpi=150); plt.close()

# ---- Fig 2: clustered heatmap ----
cols = [c for c in df.columns if c.startswith("mean_")]
g = sns.clustermap(df[cols].corr(), cmap="vlag", center=0, annot=True, fmt=".2f", annot_kws={"size":7}, figsize=(8,7), cbar_pos=(0.02,0.82,0.03,0.15))
g.fig.suptitle("Clustered correlation map: size features form one redundant block", y=1.01, fontsize=12, fontweight="bold")
g.savefig("w2figs/v2_cluster.png", dpi=150); plt.close("all")
order = [cols[i] for i in g.dendrogram_row.reordered_ind]
print("cluster order:", order)

# ---- Fig 3: PCA scatter + scree ----
X = StandardScaler().fit_transform(df[num]); p = PCA().fit(X); Z = p.transform(X)
ev = p.explained_variance_ratio_; cum = ev.cumsum()
print("PC1 %.1f PC2 %.1f cum2 %.1f; PCs for 90%%: %d; 95%%: %d" % (ev[0]*100, ev[1]*100, cum[1]*100, (cum<.90).sum()+1, (cum<.95).sum()+1))
fig, ax = plt.subplots(1,2, figsize=(12,4.8), gridspec_kw={"width_ratios":[1.4,1]})
sns.scatterplot(x=Z[:,0], y=Z[:,1], hue=df.diagnosis, palette=pal, alpha=.75, ax=ax[0], s=30)
ax[0].set_xlabel(f"PC1 ({ev[0]*100:.1f}% of variance)"); ax[0].set_ylabel(f"PC2 ({ev[1]*100:.1f}% of variance)")
ax[0].set_title("30 features squeezed into 2 dimensions"); ax[0].legend(title="")
ax[1].bar(range(1,11), ev[:10]*100, color="#9ecae1", label="Per component")
ax[1].plot(range(1,11), cum[:10]*100, "o-", color="#08519c", label="Cumulative")
ax[1].axhline(90, ls="--", c="grey"); ax[1].text(5.2, 91.5, "90% line", color="grey")
ax[1].set_xlabel("Principal component"); ax[1].set_ylabel("% variance explained"); ax[1].set_title("Scree plot"); ax[1].legend()
plt.tight_layout(); plt.savefig("w2figs/v3_pca.png", dpi=150); plt.close()
acc2 = cross_val_score(LogisticRegression(max_iter=1000), Z[:,:2], df.y, cv=5).mean()
accall = cross_val_score(LogisticRegression(max_iter=2000), X, df.y, cv=5).mean()
print("CV acc 2 PCs: %.3f ; all 30 feats: %.3f" % (acc2, accall))

# ---- Fig 4: correlation with malignancy (diverging bar) ----
r = df[list(num)+["y"]].corr()["y"].drop("y").sort_values()
sel = pd.concat([r.head(5), r.tail(8)])
plt.figure(figsize=(8,5.5))
plt.barh(sel.index.str.replace("_"," "), sel.values, color=["#1f77b4" if v<0 else "#d62728" for v in sel.values])
plt.axvline(0,c="k",lw=.8); plt.xlabel("Correlation with malignant diagnosis")
plt.title("Which measurements point toward malignancy?", fontweight="bold")
for i,v in enumerate(sel.values): plt.text(v+(0.01 if v>=0 else -0.01), i, f"{v:.2f}", va="center", ha="left" if v>=0 else "right", fontsize=8)
plt.xlim(-0.55,0.95); plt.tight_layout(); plt.savefig("w2figs/v4_bar.png", dpi=150); plt.close()
print(sel.round(2))

# ---- Fig 5: pairgrid of top features ----
top = ["worst_perimeter","worst_concave_points","mean_concavity","mean_texture"]
pg = sns.pairplot(df[top+["diagnosis"]], hue="diagnosis", palette=pal, corner=True, plot_kws={"alpha":.6,"s":18}, diag_kind="kde", height=2.1)
pg.fig.suptitle("Pairwise view: two features already separate the classes", y=1.01, fontweight="bold")
pg.savefig("w2figs/v5_pair.png", dpi=140); plt.close("all")

# ---- Fig 6: same feature, four chart types ----
fig, ax = plt.subplots(2,2, figsize=(10,7))
sns.histplot(df, x="mean_area", hue="diagnosis", palette=pal, bins=30, ax=ax[0,0]); ax[0,0].set_title("A. Histogram")
sns.kdeplot(df, x="mean_area", hue="diagnosis", palette=pal, fill=True, ax=ax[0,1]); ax[0,1].set_title("B. Density (KDE)")
sns.boxplot(df, x="diagnosis", y="mean_area", hue="diagnosis", palette=pal, ax=ax[1,0], order=["Benign","Malignant"]); ax[1,0].set_title("C. Box plot")
sns.violinplot(df, x="diagnosis", y="mean_area", hue="diagnosis", palette=pal, ax=ax[1,1], inner="point", order=["Benign","Malignant"]); ax[1,1].set_title("D. Violin plot")
fig.suptitle("Same data (mean area), four chart types", fontweight="bold"); plt.tight_layout(); plt.savefig("w2figs/v6_types.png", dpi=150); plt.close()

# ---- Fig 7: malignancy rate by area quartile ----
df["area_q"] = pd.qcut(df.mean_area, 4, labels=["Q1 smallest","Q2","Q3","Q4 largest"])
rate = df.groupby("area_q", observed=True).y.mean()*100
cnt = df.groupby("area_q", observed=True).size()
print(rate.round(1), cnt)
plt.figure(figsize=(7,4.5))
bars = plt.bar(rate.index.astype(str), rate.values, color=["#c6dbef","#9ecae1","#fb9a99","#d62728"])
for b,v in zip(bars, rate.values): plt.text(b.get_x()+b.get_width()/2, v+1.5, f"{v:.0f}%", ha="center", fontweight="bold")
plt.ylim(0,110); plt.ylabel("% of patients with malignant tumour"); plt.xlabel("Tumour size group (mean area quartile)")
plt.title("Risk climbs steeply with tumour size", fontweight="bold"); plt.tight_layout(); plt.savefig("w2figs/v7_risk.png", dpi=150); plt.close()
