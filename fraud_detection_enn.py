# =============================================================================
# ĐỒ ÁN CUỐI KỲ - KHAI PHÁ DỮ LIỆU
# CHỦ ĐỀ: Phát hiện gian lận thẻ tín dụng sử dụng thuật toán ENN
# (Edited Nearest Neighbors)
# =============================================================================
# Trường: Đại học UEH
# Môn học: Khai phá dữ liệu
# Dataset: Credit Card Fraud Detection (Kaggle)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 0: IMPORT THƯ VIỆN
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
from imblearn.under_sampling import EditedNearestNeighbours

warnings.filterwarnings('ignore')

# Thiết lập style cho biểu đồ
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
PALETTE = {"0 - Bình thường": "#2196F3", "1 - Gian lận": "#F44336"}

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("  PHÁT HIỆN GIAN LẬN THẺ TÍN DỤNG - THUẬT TOÁN ENN")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 1: TẢI DỮ LIỆU
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Đang tải dữ liệu...")

# Tải dataset từ file CSV (đặt file creditcard.csv cùng thư mục)
try:
    df = pd.read_csv("creditcard.csv")
    print(f"    ✓ Tải thành công: {df.shape[0]:,} dòng × {df.shape[1]} cột")
except FileNotFoundError:
    # Tạo dữ liệu mô phỏng nếu không có file thật
    print("    ! Không tìm thấy creditcard.csv → Tạo dữ liệu mô phỏng...")
    np.random.seed(42)
    n_normal, n_fraud = 28_000, 492
    n_total = n_normal + n_fraud

    # Tạo các feature PCA V1-V28
    normal_data = np.random.randn(n_normal, 28) * np.random.uniform(0.5, 2.5, 28)
    fraud_data  = np.random.randn(n_fraud,  28) * np.random.uniform(0.3, 1.8, 28) + \
                  np.random.uniform(-2, 2, 28)

    V_cols = [f"V{i}" for i in range(1, 29)]
    df_normal = pd.DataFrame(normal_data, columns=V_cols)
    df_fraud  = pd.DataFrame(fraud_data,  columns=V_cols)

    df_normal["Time"]   = np.random.uniform(0, 172792, n_normal)
    df_fraud["Time"]    = np.random.uniform(0, 172792, n_fraud)
    df_normal["Amount"] = np.abs(np.random.exponential(scale=90, size=n_normal))
    df_fraud["Amount"]  = np.abs(np.random.exponential(scale=130, size=n_fraud))
    df_normal["Class"]  = 0
    df_fraud["Class"]   = 1

    df = pd.concat([df_normal, df_fraud], ignore_index=True).sample(frac=1, random_state=42)
    print(f"    ✓ Dữ liệu mô phỏng: {df.shape[0]:,} dòng × {df.shape[1]} cột")

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 2: KHÁM PHÁ DỮ LIỆU (EDA)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Khám phá dữ liệu (EDA)...")

print(f"\n    Kích thước: {df.shape}")
print(f"    Kiểu dữ liệu:\n{df.dtypes.value_counts().to_string()}")
print(f"\n    Số giá trị thiếu: {df.isnull().sum().sum()}")
print(f"\n    Thống kê mô tả:\n{df.describe().round(3).to_string()}")

# Phân tích mất cân bằng
counts   = df["Class"].value_counts()
fraud_r  = counts[1] / len(df) * 100
normal_r = counts[0] / len(df) * 100
print(f"\n    Phân phối nhãn:")
print(f"      Bình thường (0): {counts[0]:,}  ({normal_r:.2f}%)")
print(f"      Gian lận   (1): {counts[1]:,}  ({fraud_r:.2f}%)")
print(f"      Tỉ lệ mất cân bằng: 1 : {counts[0]//counts[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 3: TRỰC QUAN HÓA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Tạo biểu đồ trực quan...")

# 3a. Phân phối nhãn
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Hình 1: Phân phối lớp dữ liệu (Class Distribution)", fontsize=14, fontweight='bold')

colors = ["#2196F3", "#F44336"]
axes[0].bar(["Bình thường (0)", "Gian lận (1)"], [counts[0], counts[1]],
            color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_title("Số lượng theo lớp")
axes[0].set_ylabel("Số giao dịch")
for i, v in enumerate([counts[0], counts[1]]):
    axes[0].text(i, v + 100, f"{v:,}", ha='center', fontweight='bold')

axes[1].pie([counts[0], counts[1]], labels=["Bình thường", "Gian lận"],
            colors=colors, autopct='%1.3f%%', startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title("Tỷ lệ phần trăm")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_class_distribution.png", bbox_inches='tight')
plt.close()
print("    ✓ Hình 1: Phân phối nhãn")

# 3b. Correlation heatmap (top 15 features)
fig, ax = plt.subplots(figsize=(14, 10))
top_features = df.drop(columns=["Class"]).iloc[:, :15].columns.tolist() + ["Class"]
corr = df[top_features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=.5, ax=ax,
            annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
ax.set_title("Hình 2: Ma trận tương quan (Correlation Heatmap)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_correlation_heatmap.png", bbox_inches='tight')
plt.close()
print("    ✓ Hình 2: Correlation heatmap")

# 3c. So sánh Amount giữa 2 lớp
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Hình 3: So sánh đặc trưng Amount giữa Bình thường và Gian lận",
             fontsize=13, fontweight='bold')

df_plot = df.copy()
df_plot["Nhãn"] = df_plot["Class"].map({0: "Bình thường", 1: "Gian lận"})

# Boxplot
df_plot.boxplot(column="Amount", by="Class", ax=axes[0],
                patch_artist=True, notch=False)
axes[0].set_title("Boxplot - Amount")
axes[0].set_xlabel("Lớp (0=BT, 1=GL)")
axes[0].set_ylabel("Giá trị Amount")
plt.sca(axes[0]); plt.xticks([1, 2], ["Bình thường", "Gian lận"])

# Violin plot
sns.violinplot(data=df_plot, x="Nhãn", y="Amount", palette=colors, ax=axes[1])
axes[1].set_title("Violin Plot - Amount")
axes[1].set_xlabel("")

# KDE - Amount
for cls, color, label in [(0, "#2196F3", "Bình thường"), (1, "#F44336", "Gian lận")]:
    data_sub = df[df["Class"] == cls]["Amount"]
    axes[2].hist(data_sub[data_sub < data_sub.quantile(0.99)],
                 bins=50, alpha=0.6, color=color, label=label, density=True)
axes[2].set_title("Phân phối Amount (loại ngoại lệ)")
axes[2].legend()
axes[2].set_xlabel("Amount")
axes[2].set_ylabel("Mật độ")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_amount_comparison.png", bbox_inches='tight')
plt.close()
print("    ✓ Hình 3: So sánh Amount")

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 4: TIỀN XỬ LÝ DỮ LIỆU
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Tiền xử lý dữ liệu...")

# Chuẩn hóa Amount và Time
scaler = StandardScaler()
df["scaled_Amount"] = scaler.fit_transform(df[["Amount"]])
df["scaled_Time"]   = scaler.fit_transform(df[["Time"]])

# Xây dựng tập features và nhãn
drop_cols = ["Amount", "Time", "Class"]
X = df.drop(columns=drop_cols).copy()
X["scaled_Amount"] = df["scaled_Amount"]
X["scaled_Time"]   = df["scaled_Time"]
y = df["Class"].values

print(f"    Kích thước X: {X.shape}")
print(f"    Phân phối y: {np.bincount(y)}")

# Chia train/test theo tỉ lệ 80:20, giữ stratify để duy trì tỉ lệ lớp
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Tập huấn luyện: {X_train.shape[0]:,} mẫu  |  Tập kiểm tra: {X_test.shape[0]:,} mẫu")
print(f"    Gian lận trong train: {y_train.sum()} | Gian lận trong test: {y_test.sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 5: HUẤN LUYỆN MÔ HÌNH TRƯỚC KHI ÁP DỤNG ENN (BASELINE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Huấn luyện mô hình Baseline (không dùng ENN)...")

# Sử dụng Logistic Regression và Random Forest
models_baseline = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "Decision Tree":       DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=10)
}

results_before = {}

for name, model in models_baseline.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]

    results_before[name] = {
        "model":     model,
        "y_pred":    y_pred,
        "y_prob":    y_prob,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "cm":        confusion_matrix(y_test, y_pred)
    }
    r = results_before[name]
    print(f"\n    [{name}] - BEFORE ENN")
    print(f"      Accuracy : {r['accuracy']:.4f}")
    print(f"      Precision: {r['precision']:.4f}")
    print(f"      Recall   : {r['recall']:.4f}")
    print(f"      F1-Score : {r['f1']:.4f}")
    print(f"      ROC-AUC  : {r['roc_auc']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 6: ÁP DỤNG THUẬT TOÁN ENN (EDITED NEAREST NEIGHBORS)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Áp dụng thuật toán ENN (Edited Nearest Neighbors)...")

# Khởi tạo ENN với k=3 láng giềng
enn = EditedNearestNeighbours(n_neighbors=3, kind_sel='all')
X_train_enn, y_train_enn = enn.fit_resample(X_train, y_train)

print(f"    Trước ENN: {X_train.shape[0]:,} mẫu  →  Gian lận: {y_train.sum()}")
print(f"    Sau  ENN: {X_train_enn.shape[0]:,} mẫu  →  Gian lận: {y_train_enn.sum()}")
print(f"    Đã loại: {X_train.shape[0] - X_train_enn.shape[0]:,} mẫu nhiễu")
print(f"    Tỉ lệ gian lận mới: {y_train_enn.sum()/len(y_train_enn)*100:.2f}%")

# Biểu đồ so sánh trước/sau ENN
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Hình 4: Phân phối tập huấn luyện trước và sau ENN",
             fontsize=13, fontweight='bold')

for ax, data, title in [
    (axes[0], y_train,     f"Trước ENN\n(Tổng: {len(y_train):,})"),
    (axes[1], y_train_enn, f"Sau ENN\n(Tổng: {len(y_train_enn):,})")
]:
    cnts = np.bincount(data)
    ax.bar(["Bình thường (0)", "Gian lận (1)"], cnts, color=["#2196F3", "#F44336"],
           edgecolor='white', linewidth=1.5)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Số mẫu")
    for i, v in enumerate(cnts):
        ax.text(i, v + 5, f"{v:,}", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_before_after_enn.png", bbox_inches='tight')
plt.close()
print("    ✓ Hình 4: Phân phối trước/sau ENN")

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 7: HUẤN LUYỆN MÔ HÌNH SAU KHI ÁP DỤNG ENN
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Huấn luyện mô hình sau ENN...")

models_enn = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "Decision Tree":       DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=10)
}

results_after = {}

for name, model in models_enn.items():
    model.fit(X_train_enn, y_train_enn)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results_after[name] = {
        "model":     model,
        "y_pred":    y_pred,
        "y_prob":    y_prob,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "cm":        confusion_matrix(y_test, y_pred)
    }
    r = results_after[name]
    print(f"\n    [{name}] - AFTER ENN")
    print(f"      Accuracy : {r['accuracy']:.4f}")
    print(f"      Precision: {r['precision']:.4f}")
    print(f"      Recall   : {r['recall']:.4f}")
    print(f"      F1-Score : {r['f1']:.4f}")
    print(f"      ROC-AUC  : {r['roc_auc']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 8: CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] Tạo Confusion Matrix...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Hình 5: Confusion Matrix - Trước và Sau ENN",
             fontsize=14, fontweight='bold')
axes = axes.flatten()
labels = ["Bình thường", "Gian lận"]
model_names = list(results_before.keys())

for i, name in enumerate(model_names):
    # Before ENN
    cm_b = results_before[name]["cm"]
    sns.heatmap(cm_b, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=axes[i], linewidths=1, linecolor='white',
                annot_kws={"size": 12, "weight": "bold"})
    axes[i].set_title(f"{name}\nTrước ENN", fontsize=10)
    axes[i].set_xlabel("Dự đoán"); axes[i].set_ylabel("Thực tế")

for i, name in enumerate(model_names):
    # After ENN
    cm_a = results_after[name]["cm"]
    sns.heatmap(cm_a, annot=True, fmt='d', cmap='Reds',
                xticklabels=labels, yticklabels=labels,
                ax=axes[i+3], linewidths=1, linecolor='white',
                annot_kws={"size": 12, "weight": "bold"})
    axes[i+3].set_title(f"{name}\nSau ENN", fontsize=10)
    axes[i+3].set_xlabel("Dự đoán"); axes[i+3].set_ylabel("Thực tế")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig5_confusion_matrix.png", bbox_inches='tight')
plt.close()
print("    ✓ Hình 5: Confusion Matrix")

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 9: ROC CURVE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] Vẽ ROC Curve...")

line_styles = ['-', '--', ':']
colors_roc  = ['#1976D2', '#E53935', '#2E7D32']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Hình 6: ROC Curve - Trước và Sau ENN", fontsize=14, fontweight='bold')

for ax, results, title in [
    (axes[0], results_before, "Trước ENN"),
    (axes[1], results_after,  "Sau ENN")
]:
    for i, name in enumerate(model_names):
        fpr, tpr, _ = roc_curve(y_test, results[name]["y_prob"])
        auc = results[name]["roc_auc"]
        ax.plot(fpr, tpr, lw=2, ls=line_styles[i],
                color=colors_roc[i], label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig6_roc_curve.png", bbox_inches='tight')
plt.close()
print("    ✓ Hình 6: ROC Curve")

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 10: SO SÁNH TỔNG QUAN TRƯỚC VS SAU ENN
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] Biểu đồ so sánh tổng quan Trước vs Sau ENN...")

metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

fig, axes = plt.subplots(1, len(model_names), figsize=(18, 6))
fig.suptitle("Hình 7: So sánh Hiệu suất Trước và Sau ENN theo từng mô hình",
             fontsize=13, fontweight='bold')

x = np.arange(len(metrics))
width = 0.35

for i, name in enumerate(model_names):
    vals_b = [results_before[name][m] for m in metrics]
    vals_a = [results_after[name][m]  for m in metrics]

    bars1 = axes[i].bar(x - width/2, vals_b, width, label='Trước ENN',
                        color='#2196F3', alpha=0.8, edgecolor='white')
    bars2 = axes[i].bar(x + width/2, vals_a, width, label='Sau ENN',
                        color='#F44336', alpha=0.8, edgecolor='white')

    axes[i].set_title(name, fontsize=11, fontweight='bold')
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(metric_labels, rotation=20, ha='right', fontsize=9)
    axes[i].set_ylim(0, 1.12)
    axes[i].set_ylabel("Giá trị")
    axes[i].legend(fontsize=8)
    axes[i].grid(axis='y', alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        axes[i].annotate(f'{h:.3f}',
            xy=(bar.get_x() + bar.get_width()/2, h),
            xytext=(0, 2), textcoords="offset points",
            ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        axes[i].annotate(f'{h:.3f}',
            xy=(bar.get_x() + bar.get_width()/2, h),
            xytext=(0, 2), textcoords="offset points",
            ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig7_comparison.png", bbox_inches='tight')
plt.close()
print("    ✓ Hình 7: Biểu đồ so sánh")

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 11: BẢNG TỔNG HỢP KẾT QUẢ
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  BẢNG TỔNG HỢP KẾT QUẢ")
print("=" * 65)

rows = []
for phase, results in [("Trước ENN", results_before), ("Sau ENN", results_after)]:
    for name in model_names:
        r = results[name]
        rows.append({
            "Giai đoạn": phase,
            "Mô hình":   name,
            "Accuracy":  round(r["accuracy"],  4),
            "Precision": round(r["precision"], 4),
            "Recall":    round(r["recall"],    4),
            "F1-Score":  round(r["f1"],        4),
            "ROC-AUC":   round(r["roc_auc"],   4),
        })

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(f"{OUTPUT_DIR}/summary_results.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 12: PHÂN TÍCH THAY ĐỔI
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHÂN TÍCH TÁC ĐỘNG CỦA ENN")
print("=" * 65)

for name in model_names:
    b = results_before[name]
    a = results_after[name]
    print(f"\n  {name}:")
    print(f"    Recall     : {b['recall']:.4f} → {a['recall']:.4f}  "
          f"({'↑' if a['recall']>b['recall'] else '↓'} {abs(a['recall']-b['recall'])*100:.2f}%)")
    print(f"    F1-Score   : {b['f1']:.4f} → {a['f1']:.4f}  "
          f"({'↑' if a['f1']>b['f1'] else '↓'} {abs(a['f1']-b['f1'])*100:.2f}%)")
    print(f"    ROC-AUC    : {b['roc_auc']:.4f} → {a['roc_auc']:.4f}  "
          f"({'↑' if a['roc_auc']>b['roc_auc'] else '↓'} {abs(a['roc_auc']-b['roc_auc'])*100:.2f}%)")

print("\n" + "=" * 65)
print("  HOÀN THÀNH! Kết quả lưu tại thư mục:", OUTPUT_DIR)
print("=" * 65)
