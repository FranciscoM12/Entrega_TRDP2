"""
Detección de fraude - Pipeline mejorado (TP Integrador, Taller de Resolución de Problemas II)

Cambios principales respecto al script original:
  1. LIMPIEZA: el dataset tiene 150.000 transacciones reales + 100.000 clones con ruido (todos fraude=0).
     Se deduplica por 'Unnamed: 0' (el índice original), conservando la fila original. El script anterior
     deduplicaba por (i, fecha) con keep='first', que depende del orden del archivo.
  2. SPLIT TEMPORAL Train/Validation/Test (60/20/20). Umbral elegido en VALIDATION, métricas finales en TEST.
  3. FEATURE ENGINEERING sin leakage: encodings calculados solo con train (target encoding out-of-fold),
     recuperación de 'c' desde 'r', indicadores de nulos, hora cíclica, features de texto del título, etc.
  4. MODELOS: Regresión Logística (baseline), Random Forest, HistGB, LightGBM, XGBoost, + SMOTE + Stacking/Blending.
  5. COSTOS: umbral óptimo en validación con tope duro de FP, análisis de sensibilidad (costo FP) y curva de límite de FP.
  6. ANOMALÍAS: Isolation Forest sin usar 'fraude' ni 'score', y cruce contra el fraude real y contra el modelo supervisado.
"""
import warnings
warnings.filterwarnings("ignore")
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

CSV = sys.argv[1] if len(sys.argv) > 1 else ("fraud_dataset_v2.csv" if os.path.exists("fraud_dataset_v2.csv") else "fraud_dataset_v2_2.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
SEED = 42
COSTO_FP, COSTO_FN = 5, 200
# Tope duro de FP: máximo 5% de las transacciones legítimas bloqueadas/verificadas (ver justificación en el informe)
FP_CAP_RATE = 0.01
os.makedirs(OUT, exist_ok=True)

def log(*a):
    print(*a, flush=True)

# ======================================================================================
# 1. CARGA Y LIMPIEZA
# ======================================================================================
raw = pd.read_csv(CSV)
log(f"Filas crudas: {len(raw):,}")
df = raw.drop_duplicates(subset="Unnamed: 0", keep="first").reset_index(drop=True)
log(f"Tras quitar clones (mismo índice original): {len(df):,}  | fraude={df.fraude.mean()*100:.2f}%")
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)

# ======================================================================================
# 2. SPLIT TEMPORAL 60/20/20
# ======================================================================================
n = len(df)
i1, i2 = int(n * .6), int(n * .8)
tr, va, te = df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()
for nm, d in [("train", tr), ("valid", va), ("test", te)]:
    log(f"{nm:6s}: {len(d):6,} filas | {d.fecha.min():%Y-%m-%d} -> {d.fecha.max():%Y-%m-%d} | fraude={d.fraude.mean()*100:.2f}%")
dias_test = (te.fecha.max() - te.fecha.min()).total_seconds() / 86400

# ======================================================================================
# 3. FEATURE ENGINEERING (todo lo "aprendido" se ajusta SOLO con train)
# ======================================================================================
def te_smooth(keys_fit, y_fit, keys_apply, prior, m=20):
    g = pd.DataFrame({"k": keys_fit.values, "y": y_fit.values}).groupby("k")["y"].agg(["sum", "count"])
    enc = (g["sum"] + prior * m) / (g["count"] + m)
    return keys_apply.map(enc).fillna(prior).values

class FE:
    """Ajusta estadísticas con train y transforma train/val/test sin leakage."""
    top_g = ["BR", "AR", "UY", "US"]

    def fit(self, d):
        self.prior = d.fraude.mean()
        self.j_freq = d.j.value_counts()
        self.i_freq = d.i.value_counts()
        d_ = d.assign(lm=np.log1p(d.monto))
        self.j_med = d_.groupby("j")["lm"].median()
        self.j_cnt = d_.groupby("j")["lm"].count()
        self.i_med = d_.groupby("i")["lm"].median()
        self.glob_med = d_["lm"].median()
        self.d_ref = d
        return self

    def transform(self, d, is_train=False, oof_te=None):
        X = pd.DataFrame(index=d.index)
        f = d.fecha
        # --- temporales
        X["hora"] = f.dt.hour
        X["hora_sin"] = np.sin(2 * np.pi * X.hora / 24)
        X["hora_cos"] = np.cos(2 * np.pi * X.hora / 24)
        X["dow"] = f.dt.dayofweek
        X["fin_de_semana"] = (X.dow >= 5).astype(int)
        X["madrugada"] = X.hora.between(0, 5).astype(int)      # hallazgo EDA: fraude 10-17% de 0 a 4 hs vs ~4.5% resto
        X["noche_tarde"] = X.hora.isin([22, 23]).astype(int)
        # --- monto y montos relacionados
        X["monto_log"] = np.log1p(d.monto)
        # 'c' tiene 8.6% de nulos pero 'r' ≈ int(c) siempre está -> se recupera c desde r (fila a fila, sin leakage)
        c_rec = d.c.fillna(d.r.astype(float))
        X["c_log"] = np.log1p(c_rec)
        X["r_log"] = np.log1p(d.r)
        X["monto_sobre_c"] = X.monto_log - X.c_log
        X["b"] = d.b
        # --- nulos como señal (b/c ausentes -> 6.4% fraude vs 4.9%)
        X["b_nulo"] = d.b.isna().astype(int)
        X["o_nulo"] = d.o.isna().astype(int)
        X["o_Y"] = (d.o == "Y").astype(int)
        X["p_Y"] = (d.p == "Y").astype(int)         # 's' es idéntica a p (100*p_Y) -> se descarta
        X["meta_nula"] = d[["d", "m", "q"]].isna().any(axis=1).astype(int)
        # --- numéricas originales (sentinelas -1 se dejan: los árboles las separan; para LR hay imputación)
        for c in ["a", "d", "e", "f", "h", "k", "l", "m", "n", "q"]:
            X[c] = d[c]
        X["e_log"] = np.log1p(d.e.clip(lower=0))
        X["f_log"] = np.log1p(d.f.clip(lower=0))
        X["d_neg"] = (d.d < 0).astype(int)
        X["h_neg"] = (d.h < 0).astype(int)
        X["l_neg"] = (d.l < 0).astype(int)
        X["ratio_l_m"] = d.l / (d.m.clip(lower=0) + 1)
        # --- geografía
        for gk in self.top_g:
            X[f"g_{gk}"] = (d.g == gk).astype(int)
        X["g_otro"] = (~d.g.isin(self.top_g)).astype(int)
        # --- categoría j (alta cardinalidad): frecuencia + target encoding OOF + precio relativo
        X["j_freq"] = d.j.map(self.j_freq).fillna(0)
        X["i_freq"] = d.i.map(self.i_freq).fillna(0)
        X["j_te"] = oof_te if is_train else te_smooth(self.d_ref.j, self.d_ref.fraude, d.j, self.prior)
        jm = d.j.map(self.j_med).fillna(self.glob_med)
        X["monto_vs_j"] = X.monto_log - jm
        im = d.i.map(self.i_med).fillna(self.glob_med)
        X["monto_vs_i"] = X.monto_log - im
        # --- ratios financieros y perfil
        X["monto_sobre_l"] = X.monto_log / (np.log1p(d.l.clip(lower=0)) + 1)
        X["monto_sobre_m"] = X.monto_log / (np.log1p(d.m.clip(lower=0)) + 1)
        X["monto_x_madrugada"] = X.monto_log * X.madrugada
        X["n_nulos_total"] = d.isna().sum(axis=1)
        # --- texto del título 'i'
        s = d.i.fillna("")
        X["i_len"] = s.str.len()
        X["i_words"] = s.str.split().str.len()
        X["i_upper_ratio"] = s.apply(lambda t: sum(ch.isupper() for ch in t) / (len(t) + 1))
        X["i_digit_ratio"] = s.apply(lambda t: sum(ch.isdigit() for ch in t) / (len(t) + 1))
        X["i_special"] = s.str.count(r"[^\w\s]")
        X["i_empieza_simbolo"] = s.str.match(r"^\W").astype(int)
        #nuevas
        

        return X

def oof_target_encoding(d, k=5):
    out = np.zeros(len(d))
    skf = StratifiedKFold(k, shuffle=True, random_state=SEED)
    for a, b in skf.split(d, d.fraude):
        out[b] = te_smooth(d.j.iloc[a], d.fraude.iloc[a], d.j.iloc[b], d.fraude.iloc[a].mean())
    return out

fe = FE().fit(tr)
Xtr = fe.transform(tr, is_train=True, oof_te=oof_target_encoding(tr))
Xva, Xte = fe.transform(va), fe.transform(te)


# Escala 0-1 ajustada SOLO con train (sin leakage); val/test se recortan a [0, 1]
scaler01 = MinMaxScaler(feature_range=(0, 1), clip=True).fit(Xtr)
Xtr, Xva, Xte = [pd.DataFrame(scaler01.transform(X), columns=X.columns, index=X.index)
                 for X in (Xtr, Xva, Xte)]
ytr, yva, yte = tr.fraude.values, va.fraude.values, te.fraude.values
log(f"Features: {Xtr.shape[1]}  ->", list(Xtr.columns))

# ======================================================================================
# 4. MÉTRICAS DE NEGOCIO Y UMBRAL
# ======================================================================================
def conf(y, p, t):
    pred = p >= t
    tp = int(((pred) & (y == 1)).sum()); fp = int(((pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum()); tn = int(((~pred) & (y == 0)).sum())
    return tp, fp, fn, tn

def costo(fp, fn, cfp=COSTO_FP, cfn=COSTO_FN):
    return fp * cfp + fn * cfn

def best_threshold(y, p, cfp=COSTO_FP, fp_cap=None):
    """Barre todos los umbrales (cuantiles finos del score) y elige el de mínimo costo con tope de FP."""
    cand = np.unique(np.quantile(p, np.linspace(0.5, 0.9999, 1500)))
    best = None
    for t in cand:
        tp, fp, fn, tn = conf(y, p, t)
        if fp_cap is not None and fp > fp_cap:
            continue
        c = costo(fp, fn, cfp)
        if best is None or c < best[1]:
            best = (t, c)
    return best[0] if best else cand[-1]

def evaluar(nombre, p_va, p_te, cfp=COSTO_FP, cap_rate=FP_CAP_RATE):
    cap_va = int(cap_rate * (yva == 0).sum())
    t = best_threshold(yva, p_va, cfp, cap_va)
    tp, fp, fn, tn = conf(yte, p_te, t)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    c = costo(fp, fn, cfp)
    return dict(Modelo=nombre, PR_AUC_val=average_precision_score(yva, p_va), PR_AUC_test=average_precision_score(yte, p_te),
                ROC_AUC_test=roc_auc_score(yte, p_te), umbral=t, Precision=prec, Recall=rec, TP=tp, FP=fp, FN=fn,
                Costo_test=c, Costo_mensual_est=c * 30 / dias_test)

# ======================================================================================
# 5. MODELOS
# ======================================================================================
res, preds = [], {}
def registrar(nombre, pva, pte):
    preds[nombre] = (pva, pte)
    r = evaluar(nombre, pva, pte); res.append(r)
    log(f"{nombre:34s} PR-AUC val={r['PR_AUC_val']:.4f} test={r['PR_AUC_test']:.4f} | P={r['Precision']:.3f} R={r['Recall']:.3f} "
        f"FP={r['FP']:5d} FN={r['FN']:4d} costo=${r['Costo_test']:,}")

pos_w = (ytr == 0).sum() / (ytr == 1).sum()

# 5.1 Baseline: Regresión logística
lr = make_pipeline(SimpleImputer(strategy="median"),
                   LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5))
lr.fit(Xtr, ytr)
registrar("1. Reg. Logística (baseline)", lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xte)[:, 1])

# 5.2 Regresión logística + SMOTE (solo sobre train, ya imputado/escalado)
imp_sc = make_pipeline(SimpleImputer(strategy="median")).fit(Xtr)
Xs, ys = SMOTE(sampling_strategy=0.3, random_state=SEED).fit_resample(imp_sc.transform(Xtr), ytr)
lr_sm = LogisticRegression(max_iter=2000, C=0.5).fit(Xs, ys)
registrar("2. Reg. Logística + SMOTE", lr_sm.predict_proba(imp_sc.transform(Xva))[:, 1], lr_sm.predict_proba(imp_sc.transform(Xte))[:, 1])

# 5.3 Random Forest
rf = make_pipeline(SimpleImputer(strategy="median"),
                   RandomForestClassifier(n_estimators=250, min_samples_leaf=5, max_features="sqrt",
                                          class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))
rf.fit(Xtr, ytr)
registrar("3. Random Forest", rf.predict_proba(Xva)[:, 1], rf.predict_proba(Xte)[:, 1])

# 5.4 HistGradientBoosting
hgb = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
                                     l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                     n_iter_no_change=30, random_state=SEED)
hgb.fit(Xtr, ytr)
registrar("4. HistGradientBoosting", hgb.predict_proba(Xva)[:, 1], hgb.predict_proba(Xte)[:, 1])

# 5.5 LightGBM (early stopping en validation por PR-AUC)
LGB_PARAMS = dict(n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, random_state=SEED, n_jobs=-1, verbose=-1)
lg = lgb.LGBMClassifier(**LGB_PARAMS)
lg.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
       callbacks=[lgb.early_stopping(80, verbose=False)])
log(f"   LightGBM mejor iteración: {lg.best_iteration_}")
registrar("5. LightGBM", lg.predict_proba(Xva)[:, 1], lg.predict_proba(Xte)[:, 1])

# 5.6 LightGBM + scale_pos_weight (alternativa a SMOTE)
lgw = lgb.LGBMClassifier(**{**LGB_PARAMS, "scale_pos_weight": np.sqrt(pos_w)})
lgw.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
        callbacks=[lgb.early_stopping(80, verbose=False)])
registrar("6. LightGBM + pesos de clase", lgw.predict_proba(Xva)[:, 1], lgw.predict_proba(Xte)[:, 1])

# 5.7 LightGBM + SMOTE
imp = SimpleImputer(strategy="median").fit(Xtr)
Xs2, ys2 = SMOTE(sampling_strategy=0.2, random_state=SEED).fit_resample(imp.transform(Xtr), ytr)
lgs = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": 600})
lgs.fit(Xs2, ys2)
registrar("7. LightGBM + SMOTE", lgs.predict_proba(imp.transform(Xva))[:, 1], lgs.predict_proba(imp.transform(Xte))[:, 1])

# 5.8 XGBoost
xg = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.03, max_depth=5, min_child_weight=5, subsample=0.8,
                       colsample_bytree=0.7, reg_lambda=2.0, eval_metric="aucpr", early_stopping_rounds=80,
                       tree_method="hist", n_jobs=-1, random_state=SEED)
xg.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
registrar("8. XGBoost", xg.predict_proba(Xva)[:, 1], xg.predict_proba(Xte)[:, 1])

# 5.9 CatBoost
cb_model = cb.CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                                 eval_metric="PRAUC", random_seed=SEED, early_stopping_rounds=80,
                                 auto_class_weights="Balanced", thread_count=-1, verbose=False)
cb_model.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=False)
registrar("9. CatBoost", cb_model.predict_proba(Xva)[:, 1], cb_model.predict_proba(Xte)[:, 1])

# 5.10 Ensambles: Blending Top 3, promedio de rangos y stacking
p_b3_va = (preds["5. LightGBM"][0] + preds["8. XGBoost"][0] + preds["9. CatBoost"][0]) / 3
p_b3_te = (preds["5. LightGBM"][1] + preds["8. XGBoost"][1] + preds["9. CatBoost"][1]) / 3
registrar("10. Blend Top 3 (LGB+XGB+Cat)", p_b3_va, p_b3_te)

from scipy.stats import rankdata
def rk(v, ref):  # rango relativo respecto de la distribución de validación (evita mirar test)
    return np.searchsorted(np.sort(ref), v) / len(ref)

base = ["5. LightGBM", "8. XGBoost", "9. CatBoost", "4. HistGradientBoosting", "3. Random Forest", "1. Reg. Logística (baseline)"]
avg_va = np.mean([rk(preds[b][0], preds[b][0]) for b in base], axis=0)
avg_te = np.mean([rk(preds[b][1], preds[b][0]) for b in base], axis=0)
registrar("11. Promedio de modelos (rank avg)", avg_va, avg_te)

# Stacking: meta-learner (LR) entrenado sobre predicciones OOF de train (5 folds) de cada modelo base
log("Stacking: generando predicciones OOF en train...")
def mk_base():
    return {
        "lgb": lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": max(lg.best_iteration_, 100), "n_jobs": -1}),
        "xgb": xgb.XGBClassifier(n_estimators=max(xg.best_iteration + 1, 100), learning_rate=0.03, max_depth=5,
                                 min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
                                 tree_method="hist", n_jobs=-1, random_state=SEED),
        "cat": cb.CatBoostClassifier(iterations=max(cb_model.get_best_iteration(), 80), learning_rate=0.03,
                                     depth=6, l2_leaf_reg=3.0, auto_class_weights="Balanced",
                                     thread_count=-1, random_seed=SEED, verbose=False),
        "hgb": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=max(hgb.n_iter_, 50), max_leaf_nodes=31,
                                              min_samples_leaf=40, l2_regularization=1.0, random_state=SEED),
        "rf": make_pipeline(SimpleImputer(strategy="median"),
                            RandomForestClassifier(n_estimators=150, min_samples_leaf=5, class_weight="balanced_subsample",
                                                   n_jobs=-1, random_state=SEED)),
        "lr": make_pipeline(SimpleImputer(strategy="median"),
                            LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5)),
    }
names = list(mk_base().keys())
oof = np.zeros((len(Xtr), len(names)))
for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr):
    for k, (nm, m) in enumerate(mk_base().items()):
        m.fit(Xtr.iloc[a], ytr[a]); oof[b, k] = m.predict_proba(Xtr.iloc[b])[:, 1]
full = mk_base()
for m in full.values():
    m.fit(Xtr, ytr)
def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
meta = LogisticRegression(C=1.0, max_iter=1000).fit(logit(oof), ytr)
sva = meta.predict_proba(logit(np.column_stack([m.predict_proba(Xva)[:, 1] for m in full.values()])))[:, 1]
ste = meta.predict_proba(logit(np.column_stack([m.predict_proba(Xte)[:, 1] for m in full.values()])))[:, 1]
log("   pesos del meta-modelo:", dict(zip(names, meta.coef_[0].round(3))))
registrar("12. Stacking (meta LR)", sva, ste)

df_res = pd.DataFrame(res)
df_res.to_csv(f"{OUT}/resultados_modelos.csv", index=False)

# ======================================================================================
# 6. ELECCIÓN DEL MODELO Y ANÁLISIS DE NEGOCIO
# ======================================================================================
champ = df_res.sort_values(["Costo_test"]).iloc[0]["Modelo"]
# elegimos por PR-AUC de VALIDACIÓN (no mirar test para elegir), y reportamos costo en test
champ = df_res.sort_values("PR_AUC_val", ascending=False).iloc[0]["Modelo"]
log(f"\nModelo elegido por PR-AUC en validación: {champ}")
pva, pte = preds[champ]

log("\n--- Sensibilidad al costo de FP (umbral re-optimizado en validación, evaluado en test) ---")
sens = []
for cfp in [5, 10, 20, 50, 100]:
    r = evaluar(champ, pva, pte, cfp=cfp)
    sens.append(dict(costo_FP=cfp, umbral=r["umbral"], FP=r["FP"], FN=r["FN"], Recall=r["Recall"], Precision=r["Precision"],
                     Costo_total=r["Costo_test"]))
df_sens = pd.DataFrame(sens); log(df_sens.round(4).to_string(index=False))
df_sens.to_csv(f"{OUT}/sensibilidad_costo_fp.csv", index=False)

log("\n--- Tope duro de FP (% de legítimos afectados) ---")
caps = []
for cr in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, None]:
    cap_va = None if cr is None else int(cr * (yva == 0).sum())
    t = best_threshold(yva, pva, COSTO_FP, cap_va)
    tp, fp, fn, tn = conf(yte, pte, t)
    caps.append(dict(tope_FP_pct_legit="sin tope" if cr is None else f"{cr*100:.1f}%", FP=fp, FN=fn, TP=tp,
                     Recall=tp / (tp + fn), Precision=tp / max(tp + fp, 1), Costo=costo(fp, fn),
                     FP_mensual_est=fp * 30 / dias_test))
df_caps = pd.DataFrame(caps); log(df_caps.round(3).to_string(index=False))
df_caps.to_csv(f"{OUT}/limite_fp.csv", index=False)

# Comparación con el sistema actual (columna 'score' del dataset), mismo criterio de umbral
sc_va, sc_te = va.score.fillna(0).values, te.score.fillna(0).values
r = evaluar("Baseline negocio: columna score", sc_va, sc_te)
log(f"\nColumna 'score' sola como regla: PR-AUC test={r['PR_AUC_test']:.4f} Recall={r['Recall']:.3f} FP={r['FP']} FN={r['FN']} costo=${r['Costo_test']:,}")

# Importancia de variables (gain) del LightGBM
imp_df = pd.DataFrame({"feature": Xtr.columns, "gain": lg.booster_.feature_importance("gain")}).sort_values("gain", ascending=False)
imp_df["gain_pct"] = imp_df.gain / imp_df.gain.sum() * 100
imp_df.to_csv(f"{OUT}/importancia_variables.csv", index=False)
log("\nTop 12 variables (LightGBM, gain %):"); log(imp_df.head(12)[["feature", "gain_pct"]].round(2).to_string(index=False))

# Experimento: ¿aporta la columna score como feature?
Xtr_s, Xva_s, Xte_s = Xtr.assign(score=tr.score.values), Xva.assign(score=va.score.values), Xte.assign(score=te.score.values)
lgsc = lgb.LGBMClassifier(**LGB_PARAMS).fit(Xtr_s, ytr, eval_set=[(Xva_s, yva)], eval_metric="average_precision",
                                              callbacks=[lgb.early_stopping(80, verbose=False)])
r = evaluar("LightGBM + score como feature", lgsc.predict_proba(Xva_s)[:, 1], lgsc.predict_proba(Xte_s)[:, 1])
log(f"LightGBM + score como feature: PR-AUC test={r['PR_AUC_test']:.4f} Recall={r['Recall']:.3f} FP={r['FP']} FN={r['FN']} costo=${r['Costo_test']:,}")

# ======================================================================================
# 7. DETECCIÓN DE ANOMALÍAS (sin 'fraude' ni 'score')
# ======================================================================================
log("\n--- Isolation Forest (no supervisado) ---")
cols_anom = [c for c in Xtr.columns if c not in ("j_te",)]  # j_te usa la etiqueta -> fuera
imp_a = make_pipeline(SimpleImputer(strategy="median")).fit(Xtr[cols_anom])
Atr, Ate = imp_a.transform(Xtr[cols_anom]), imp_a.transform(Xte[cols_anom])
iso = IsolationForest(n_estimators=300, max_samples=0.5, contamination="auto", random_state=SEED, n_jobs=-1).fit(Atr)
an_te = -iso.score_samples(Ate)  # mayor = más anómalo
log(f"PR-AUC de Isolation Forest vs fraude (test): {average_precision_score(yte, an_te):.4f}  (azar = {yte.mean():.4f}) | ROC-AUC {roc_auc_score(yte, an_te):.4f}")
top_n = int(0.03 * len(te))
idx_top = np.argsort(-an_te)[:top_n]
hit = yte[idx_top].sum()
log(f"Top 3% más anómalo ({top_n} casos): {hit} son fraude reales ({hit/top_n*100:.1f}% vs base {yte.mean()*100:.1f}%) | lift x{(hit/top_n)/yte.mean():.2f}")

# cruce con modelo supervisado
t_sup = best_threshold(yva, pva, COSTO_FP, int(FP_CAP_RATE * (yva == 0).sum()))
flag_sup = pte >= t_sup
flag_an = np.zeros(len(te), bool); flag_an[idx_top] = True
cand = te.assign(anomaly_score=an_te, p_modelo=pte, flag_modelo=flag_sup, flag_anomalia=flag_an)
nuevos = cand[(cand.flag_anomalia) & (~cand.flag_modelo) & (cand.fraude == 0)].sort_values("anomaly_score", ascending=False)
log(f"Anómalos NO marcados por el modelo supervisado y etiquetados fraude=0 -> candidatos a revisar manualmente: {len(nuevos)}")
cols_show = ["fecha", "i", "j", "g", "monto", "hora"] if "hora" in cand else ["fecha", "i", "j", "g", "monto"]
nuevos[["fecha", "i", "j", "g", "monto", "anomaly_score", "p_modelo"]].head(50).to_csv(f"{OUT}/casos_a_revisar.csv", index=False)
log(pd.crosstab(cand.flag_anomalia, cand.fraude, rownames=["anomalía"], colnames=["fraude real"]).to_string())
log(pd.crosstab(cand.flag_modelo, cand.fraude, rownames=["modelo"], colnames=["fraude real"]).to_string())

# Qué distingue a los anómalos puros (perfil comparado contra el resto)
perfil = pd.DataFrame({"anomalos_no_marcados": Xte.loc[nuevos.index, ["monto_log", "hora", "madrugada", "o_nulo", "p_Y", "b_nulo", "n", "j_freq"]].mean(),
                       "resto": Xte.drop(nuevos.index)[["monto_log", "hora", "madrugada", "o_nulo", "p_Y", "b_nulo", "n", "j_freq"]].mean()})
log(perfil.round(3).to_string())
perfil.to_csv(f"{OUT}/perfil_anomalos.csv")

# Guardar predicciones del campeón
pd.DataFrame({"fecha": te.fecha.values, "fraude": yte, "p_modelo": pte, "anomaly_score": an_te}).to_csv(f"{OUT}/predicciones_test.csv", index=False)
log("\nListo. Archivos guardados en", OUT)