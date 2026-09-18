import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# ---------------- Settings ----------------
D1_PKL = "D1.pkl"
D2_PKL = "D2.pkl"
OUT_CSV = "D2_labeled.csv"
LABEL = "Label"
CAP_PER_CLASS = 200_000            # max training rows per class (None = no cap)
MIN_ROWS_PER_CLASS = 5             # classes with fewer rows are dropped
USE_BALANCED_WEIGHTS = False       # try True if rare classes have poor recall
CONF_THRESHOLD = 0.70              # D2 rows below this are flagged for review
DROP_COLUMNS = []                  # e.g. ["Unnamed: 0", "Src IP", "Dst IP", "Timestamp"]
SEED = 42

CLASS_NAMES = [
    "analysis", "backdoor", "benign", "bot", "ddos", "dos", "dos goldeneye",
    "dos hulk", "dos slowhttptest", "dos slowloris", "drdos_dns", "drdos_ldap",
    "drdos_mssql", "drdos_netbios", "drdos_ntp3", "drdos_snmp", "drdos_ssdp",
    "drdos_udp", "exploits", "ftp-patator", "fuzzers", "generic", "infiltration",
    "mssql", "portscan", "reconnaissance", "shellcode", "ssh-patator", "syn",
    "tftp", "udp-lag", "web attack - brute force", "web attack - sql injection",
    "web attack - xss", "webddos", "worms",
]


def norm_label(s):
    return s.astype(str).str.strip().str.lower().str.replace("\u2013", "-", regex=False)


CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


# ---------------- Loading helper ----------------
def load_pkl(path):
    obj = pd.read_pickle(path)
    if not isinstance(obj, pd.DataFrame):
        obj = pd.DataFrame(obj)
    obj.columns = obj.columns.astype(str).str.strip()

    for c in obj.columns:
        if c == LABEL:
            continue
        if not pd.api.types.is_numeric_dtype(obj[c]):
            s = obj[c].astype(str).str.strip()
            # numbers stored as text, possibly with thousands separators ("51,236")
            num = pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")
            if num.notna().mean() > 0.95:
                obj[c] = num
            else:
                obj[c] = s.str.upper()          # true categorical, e.g. Protocol

    obj = obj.replace([np.inf, -np.inf], np.nan)
    num_cols = obj.select_dtypes("number").columns.drop(LABEL, errors="ignore")
    obj[num_cols] = obj[num_cols].astype("float32")
    return obj


# ---------------- Load D1 and D2 ----------------
d1 = load_pkl(D1_PKL)
d2 = load_pkl(D2_PKL)

d1[LABEL] = norm_label(d1[LABEL])
n0 = len(d1)
d1 = d1.drop_duplicates().reset_index(drop=True)
print(f"D1 rows: {n0:,} -> {len(d1):,} after dropping duplicates")

counts = d1[LABEL].value_counts()
print("\nClass counts:\n", counts.to_string())

absent = [c for c in CLASS_NAMES if c not in counts.index]
if absent:
    print(f"\nNOTE: classes in the mapping but absent from D1 (cannot be predicted): {absent}")
unknown = [c for c in counts.index if c not in CLASS_ID]
if unknown:
    print(f"\nNOTE: D1 labels not in the mapping (check spelling): {unknown}")

small = counts[counts < MIN_ROWS_PER_CLASS].index
if len(small):
    print(f"\nWARNING: dropping classes with < {MIN_ROWS_PER_CLASS} rows: {list(small)}")
    d1 = d1[~d1[LABEL].isin(small)].reset_index(drop=True)

features = [c for c in d1.columns if c != LABEL and c not in DROP_COLUMNS]
missing = [c for c in features if c not in d2.columns]
if missing:
    raise ValueError(f"D2 is missing columns: {missing}")

# Categorical columns (e.g. Protocol): same category list in D1 and D2
cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(d1[c])]
for c in cat_cols:
    cats = sorted(set(d1[c].dropna().astype(str)) | set(d2[c].dropna().astype(str)))
    d1[c] = pd.Categorical(d1[c].astype(str), categories=cats)
    d2[c] = pd.Categorical(d2[c].astype(str), categories=cats)
print(f"Categorical columns: {cat_cols}")

le = LabelEncoder()
y = le.fit_transform(d1[LABEL])
X = d1[features]
n_classes = len(le.classes_)
print(f"\nFeatures: {len(features)} | Classes: {n_classes}")


# ---------------- Optional per-class cap (applied to training data only) ----------------
def cap_indices(idx, labels, cap, seed):
    if cap is None:
        return idx
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(labels[idx]):
        cls_idx = idx[labels[idx] == cls]
        if len(cls_idx) > cap:
            cls_idx = rng.choice(cls_idx, size=cap, replace=False)
        keep.append(cls_idx)
    return np.concatenate(keep)


# ---------------- Holdout split (natural class distribution in validation) ----------------
all_idx = np.arange(len(y))
tr_idx, va_idx = train_test_split(
    all_idx, test_size=0.2, stratify=y, random_state=SEED
)
tr_idx_capped = cap_indices(tr_idx, y, CAP_PER_CLASS, SEED)
print(f"Train rows: {len(tr_idx):,} (after cap: {len(tr_idx_capped):,}) | Validation rows: {len(va_idx):,}")

params = dict(
    objective="multiclass",
    num_class=n_classes,
    learning_rate=0.05,
    n_estimators=1000,                 # early stopping will cut this down
    num_leaves=63,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    class_weight="balanced" if USE_BALANCED_WEIGHTS else None,
    n_jobs=-1,
    random_state=SEED,
    verbose=-1,
)

model = lgb.LGBMClassifier(**params)
model.fit(
    X.iloc[tr_idx_capped], y[tr_idx_capped],
    eval_set=[(X.iloc[va_idx], y[va_idx])],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
)
best_iter = model.best_iteration_ or params["n_estimators"]
print(f"\nBest iteration: {best_iter}")


# ---------------- Evaluate ----------------
va_pred = model.predict(X.iloc[va_idx])
print("\nMacro-F1:", round(f1_score(y[va_idx], va_pred, average="macro"), 4))
print("\n", classification_report(
    y[va_idx], va_pred, labels=np.arange(n_classes),
    target_names=le.classes_, zero_division=0
))

cm = pd.DataFrame(
    confusion_matrix(y[va_idx], va_pred, labels=np.arange(n_classes)),
    index=le.classes_, columns=le.classes_,
)
cm.to_csv("confusion_matrix.csv")

off = cm.where(~np.eye(n_classes, dtype=bool)).stack()
print("\nTop confused class pairs (true -> predicted):")
print(off.sort_values(ascending=False).head(15).to_string())


# ---------------- Final model on all of D1 ----------------
final_idx = cap_indices(all_idx, y, CAP_PER_CLASS, SEED)
final_params = {**params, "n_estimators": best_iter}
final = lgb.LGBMClassifier(**final_params)
final.fit(X.iloc[final_idx], y[final_idx])


# ---------------- Predict D2 ----------------
proba = final.predict_proba(d2[features])
d2[LABEL] = le.inverse_transform(proba.argmax(axis=1))
d2["LabelID"] = d2[LABEL].map(CLASS_ID).fillna(-1).astype(int)
d2["Confidence"] = proba.max(axis=1)
d2["LowConfidence"] = d2["Confidence"] < CONF_THRESHOLD
d2.to_csv(OUT_CSV, index=False)

print(f"\nSaved {OUT_CSV}")
print(f"Low-confidence rows: {d2['LowConfidence'].sum():,} of {len(d2):,}")
print("\nPredicted label distribution in D2:")
print(d2[LABEL].value_counts().to_string())

# ---------------- Write output.txt (one class id per D2 row) ----------------
if (d2["LabelID"] < 0).any():
    bad = d2.loc[d2["LabelID"] < 0, LABEL].unique()
    raise ValueError(f"Predicted labels missing from the class mapping: {bad}")

with open("output.txt", "w") as f:
    f.write("\n".join(str(i) for i in d2["LabelID"]) + "\n")

print(f"Wrote output.txt with {len(d2):,} lines")
