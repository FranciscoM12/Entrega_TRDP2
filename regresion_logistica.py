"""
==========================================================================
MODELO BASE PARA PREDECIR FRAUDE — fraud_dataset_v2.csv
==========================================================================
 
Nota importante: 'fraude' es una variable binaria (0/1), así que el modelo
correcto es una REGRESIÓN LOGÍSTICA (no lineal). La logística te da una
probabilidad de fraude entre 0 y 1, y sobre esa probabilidad se aplica el
UMBRAL para decidir la clase final.
 
Este script incluye tres formas de elegir el umbral:
  1. "manual"  -> vos lo fijás a mano (UMBRAL_MANUAL)
  2. "f1"      -> el que maximiza el F1-score
  3. "costo"   -> el que minimiza el costo de negocio (ver más abajo)
 
Todo lo que normalmente cambiarías está agrupado en la sección CONFIG.
==========================================================================
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    precision_recall_curve, f1_score, average_precision_score
)
 
# ==========================================================================
# CONFIG — lo único que deberías necesitar tocar
# ==========================================================================

RUTA_CSV = "/Users/franmonti/Documents/Austral/Programacion2/Files/fraud_dataset_v2.csv"
RUTA_SALIDA_GRAFICOS = "/Users/franmonti/Documents/Austral/taller_de_problemas_2/outputs/modelo_fraude_resultados.png"


FEATURES = [
    "score_normalizado",      # score corregido por el quiebre de calibración
    #"log_monto",           #monto mejor que log monto
    "monto",               # en vez de 'monto' crudo
    "rango_horario",           # categórica (MEJORA BASTANTE)
    "falta_bc",                # ya es 0/1
    # "target_encoding_j",       # tasa de fraude histórica de la categoría (solo con train)
    # "desvio_precio_categoria", # cuánto se aleja el monto de la mediana de su categoría j
    # "desvio_precio_producto",  # cuánto se aleja el monto de la mediana de su producto i
   # "a",
    "a_cat",
    #"b",
    "c",
    "s",
     "n",
    "q",
   # "o",   # categórica Y/N/NA. (variable muuuuy imoortante para detectar fraude)
    "p",   # categórica Y/N
    # "g",   # categórica (país)
     "l",
     "m",
    "f",
    "h",
   "sin_hora",                 # Componente cíclica seno de la hora (0-23)
    "cos_hora",                 # Componente cíclica coseno de la hora (0-23)
    "es_madrugada", 
  "o_cat",                    # Categoría de 'o' incluyendo '__NA__' explícito
     "g_agrupado",               # Países principales (BR, AR, UY, US) y 'OTRO'
     "log_l",                    # log1p de la antigüedad del vendedor/cuenta (corr: -0.180)
     "falta_l",                  # Bandera de valor faltante en l
    "log_m",                    # log1p de volumen de ventas/reputación (corr: -0.135)
     "falta_m",                  # Bandera de valor faltante en m
    "log_f",                    # log1p del volumen transaccional recortado a >= 0
    "f_es_negativo",  

]
 
FEATURES_CATEGORICAS = ["o", "p", "g", "rango_horario", "a_cat", "g_agrupado", "o_cat"]
 
# Suavizado del target encoding de 'j': muchas categorías tienen muy pocas
# publicaciones (hasta 1 sola), así que sin suavizar el encoding memorizaría
# ruido. La fórmula pondera hacia el promedio global cuando hay pocos casos:
#   encoded = (n*media_categoria + SUAVIZADO*media_global) / (n + SUAVIZADO)
TARGET_ENCODING_SUAVIZADO = 20
 
# --- Umbral ---
# METODO_UMBRAL define qué umbral se usa para la evaluación final:
#   "manual"     -> usa UMBRAL_MANUAL tal cual
#   "f1"         -> usa el umbral que maximiza el F1-score
#   "costo"      -> usa el umbral que minimiza el costo de negocio (ver abajo)
#   "fpr_maximo" -> usa el umbral que da el mayor recall (detecta más fraude)
#                   sin superar un % máximo de falsos positivos permitido
METODO_UMBRAL = "f1"
UMBRAL_MANUAL = 0.5
 
# Máximo porcentaje de falsos positivos tolerado, como fracción de las
# publicaciones NO fraudulentas (usado solo si METODO_UMBRAL == "fpr_maximo").
# Ej: 0.05 = "como máximo, marcar mal al 5% de las publicaciones legítimas".
FPR_MAXIMO = 0.01
 
# --- Costo de negocio (usado solo si METODO_UMBRAL == "costo") ---
# Costo de un Falso Negativo = el monto de esa transacción fraudulenta
# que no detectamos (plata que la empresa pierde directamente).
#
# Costo de un Falso Positivo = dos partes:
#   (a) costo de reputación, que CRECE EXPONENCIALMENTE con la cantidad
#       de falsos positivos acumulados (no es lo mismo 5 que 500 clientes
#       mal marcados: el daño de reputación se acelera).
#       formula: COSTO_BASE_REPUTACION * (exp(TASA_CRECIMIENTO * n_fp) - 1)
#   (b) COSTO_REVISION_MANUAL dólares fijos por cada falso positivo,
#       por el tiempo de una persona revisando manualmente el producto.
#
# Estos dos parámetros (base y tasa) son los que vos calibrás según qué
# tan grave es el daño reputacional real para tu negocio. Están seteados
# en valores de referencia — ajustalos con datos reales si los tenés.
COSTO_REVISION_MANUAL = 7.0
COSTO_BASE_REPUTACION = 10.0
TASA_CRECIMIENTO_REPUTACION = 0.01
 
TEST_SIZE = 0.25
RANDOM_STATE = 42
 
ELIMINAR_CLONES = True
CORREGIR_CENTINELA_R = True
 
MODEL_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}
 
# ==========================================================================
# FUNCIONES DE BÚSQUEDA DE UMBRAL
# ==========================================================================
 
def buscar_umbral_max_f1(y_true, proba, n_pasos=199):
    """
    Recorre umbrales entre 0 y 1 y devuelve el que maximiza el F1-score
    (balance entre precision y recall).
    """
    umbrales = np.linspace(0.005, 0.995, n_pasos)
    f1_scores = np.array([
        f1_score(y_true, (proba >= u).astype(int), zero_division=0)
        for u in umbrales
    ])
    mejor_idx = np.argmax(f1_scores)
    return umbrales[mejor_idx], umbrales, f1_scores
 
 
def costo_reputacion(n_fp, costo_base=COSTO_BASE_REPUTACION,
                      tasa=TASA_CRECIMIENTO_REPUTACION):
    """Costo de reputación: crece exponencialmente con la cantidad de FP."""
    return costo_base * (np.exp(tasa * n_fp) - 1)
 
 
def calcular_costo_total(y_true, proba, montos, umbral,
                          costo_revision=COSTO_REVISION_MANUAL,
                          costo_base_rep=COSTO_BASE_REPUTACION,
                          tasa_rep=TASA_CRECIMIENTO_REPUTACION):
    """
    Costo total de negocio para un umbral dado:
      - Falsos negativos: suma del monto de esas transacciones no detectadas.
      - Falsos positivos: costo de reputación (exponencial en n_fp) +
        costo fijo de revisión manual por cada uno.
    Devuelve (costo_total, costo_fn, costo_fp, n_fn, n_fp).
    """
    pred = (proba >= umbral).astype(int)
    y_true = np.asarray(y_true)
    montos = np.asarray(montos)
 
    fn_mask = (y_true == 1) & (pred == 0)
    fp_mask = (y_true == 0) & (pred == 1)
 
    n_fn = fn_mask.sum()
    n_fp = fp_mask.sum()
 
    costo_fn = montos[fn_mask].sum()
    costo_fp = costo_reputacion(n_fp, costo_base_rep, tasa_rep) + costo_revision * n_fp
 
    return costo_fn + costo_fp, costo_fn, costo_fp, n_fn, n_fp
 
 
def buscar_umbral_costo_minimo(y_true, proba, montos, n_pasos=199, **kwargs):
    """
    Recorre umbrales entre 0 y 1 y devuelve el que minimiza el costo total
    de negocio (ver calcular_costo_total).
    """
    umbrales = np.linspace(0.005, 0.995, n_pasos)
    costos = np.array([
        calcular_costo_total(y_true, proba, montos, u, **kwargs)[0]
        for u in umbrales
    ])
    mejor_idx = np.argmin(costos)
    return umbrales[mejor_idx], umbrales, costos
 
 
def buscar_umbral_fpr_maximo(y_true, proba, fpr_maximo):
    """
    Busca, sobre la curva ROC real (no un barrido aproximado), el umbral que
    da el mayor recall (tasa de verdaderos positivos) sin que la tasa de
    falsos positivos supere 'fpr_maximo'.
 
    Como en la curva ROC el recall crece junto con el FPR a medida que baja
    el umbral, el punto óptimo es siempre el de mayor FPR permitido dentro
    del presupuesto — por eso alcanza con quedarse con el último punto que
    todavía cumple la restricción.
 
    Devuelve (umbral, fpr_logrado, tpr_logrado, fpr_curva, tpr_curva, umbrales_curva).
    """
    fpr_curva, tpr_curva, umbrales_curva = roc_curve(y_true, proba)
 
    cumple = fpr_curva <= fpr_maximo
    if not cumple.any():
        # Ni el umbral más estricto (fpr=0) alcanza — devolvemos ese, es
        # lo más cerca del presupuesto que se puede llegar.
        idx = 0
    else:
        idx_candidatos = np.where(cumple)[0]
        idx = idx_candidatos[np.argmax(tpr_curva[idx_candidatos])]
 
    umbral = umbrales_curva[idx]
    # El primer umbral que devuelve sklearn puede ser > 1 (un valor
    # centinela); lo recortamos a 1 para que sea un umbral usable.
    umbral = min(umbral, 1.0)
 
    return umbral, fpr_curva[idx], tpr_curva[idx], fpr_curva, tpr_curva, umbrales_curva
 
 
def target_encoding_suavizado(df_train, columna, target="fraude", suavizado=20):
    """
    Calcula un mapeo {categoria -> tasa de fraude suavizada} usando SOLO
    df_train, para evitar leakage. Categorías con pocas observaciones se
    "empujan" hacia el promedio global; categorías con muchas observaciones
    quedan dominadas por su propia tasa real.
    Devuelve (mapeo, media_global) — media_global se usa como fallback para
    categorías del test que no existan en train.
    """
    media_global = df_train[target].mean()
    stats = df_train.groupby(columna)[target].agg(["mean", "count"])
    encoded = (stats["count"] * stats["mean"] + suavizado * media_global) / (
        stats["count"] + suavizado
    )
    return encoded, media_global
 
 
# ==========================================================================
# 1. CARGA
# ==========================================================================
 
df = pd.read_csv(RUTA_CSV)
df["fecha"] = pd.to_datetime(df["fecha"])
print(f"Filas cargadas: {len(df):,}")
 
# ==========================================================================
# 2. LIMPIEZA
# ==========================================================================
 
if CORREGIR_CENTINELA_R:
    centinela_detectado = df.loc[df["c"].isna(), "r"].mode()
    if len(centinela_detectado) > 0:
        valor_centinela = centinela_detectado.iloc[0]
        n_afectadas = (df["r"] == valor_centinela).sum()
        df["r"] = df["r"].replace(valor_centinela, np.nan)
        print(f"Corregido centinela en 'r' (valor {valor_centinela}): "
              f"{n_afectadas:,} filas pasadas a NA")
 
if ELIMINAR_CLONES:
    filas_antes = len(df)
    df = df.sort_index().drop_duplicates(subset=["i", "fecha"], keep="first")
    print(f"Clones eliminados: {filas_antes - len(df):,} filas "
          f"({len(df):,} filas restantes)")
 
print(f"Tasa de fraude tras limpieza: {df['fraude'].mean()*100:.2f}%")
 
# ==========================================================================
# 2.3 — FEATURE ENGINEERING
# ==========================================================================
 
# --- (a) Corregir el quiebre de nivel del 'score' a partir del 29/03 ---
# El score promedio cae de ~51 a ~44 desde el 29/03/2020, pero la tasa de
# fraude NO cambia en ese quiebre (2.95% antes vs 3.05% después) — es decir,
# es un cambio de calibración/escala del score, no un cambio real de riesgo.
# Si usamos 'score' crudo, el modelo puede confundir "está en el período
# post-quiebre" con "es más riesgoso". Para evitarlo, generamos:
#   - una bandera del período (antes/después del quiebre)
#   - un score normalizado dentro de cada período (z-score por período),
#     que sí es comparable entre ambas mitades del dataset.
FECHA_QUIEBRE_SCORE = pd.Timestamp("2020-03-29")
 
df["periodo_score"] = np.where(df["fecha"] < FECHA_QUIEBRE_SCORE, "antes", "despues")

df["log_l"] = np.log1p(np.clip(df["l"], 0, None))
df["falta_l"] = df["l"].isna().astype(int)

df["log_m"] = np.log1p(np.clip(df["m"], 0, None))
df["falta_m"] = df["m"].isna().astype(int)

df["log_f"] = np.log1p(np.clip(df["f"], 0, None))
df["f_es_negativo"] = (df["f"] < 0).astype(int)

df["log_h"] = np.log1p(np.clip(df["h"], 0, None))
 
df["a_cat"] = df["a"].astype(str)
top_paises = ["BR", "AR", "UY", "US"]
df["g_agrupado"] = df["g"].apply(lambda x: x if x in top_paises else ("__NA__" if pd.isna(x) else "OTRO"))
df["o_cat"] = df["o"].fillna("__NA__")

df["hora"] = df["fecha"].dt.hour
df["sin_hora"] = np.sin(2 * np.pi * df["hora"] / 24.0)
df["cos_hora"] = np.cos(2 * np.pi * df["hora"] / 24.0)
df["es_madrugada"] = df["hora"].isin([0, 1, 2, 3, 4]).astype(int)

medias_periodo = df.groupby("periodo_score")["score"].transform("mean")
stds_periodo = df.groupby("periodo_score")["score"].transform("std")
df["score_normalizado"] = (df["score"] - medias_periodo) / stds_periodo
 
# --- (b) log(monto) ---
# 'monto' suele tener distribución muy asimétrica (pocos valores muy altos);
# el logaritmo la comprime y suele ayudar a modelos lineales. Usamos log1p
# (log(1+x)) para poder manejar montos de 0 sin error.
df["log_monto"] = np.log1p(df["monto"])
 
# --- (c) Rango horario ---
# Extraemos la hora de 'fecha' y la agrupamos en franjas.
df["hora"] = df["fecha"].dt.hour
df["rango_horario"] = pd.cut(
    df["hora"],
    bins=[-1, 5, 11, 17, 23],
    labels=["madrugada", "manana", "tarde", "noche"]
)
 
# --- (d) Indicador de si b/c son NA ---
# Ya vimos que b y c son NA exactamente en las mismas filas (el mismo grupo
# de ~21,474 casos), así que un solo indicador alcanza para las dos.
df["falta_bc"] = df["b"].isna().astype(int)
 
print("\nFeature engineering (parte 1 — no depende de train/test) aplicado: "
      "periodo_score, score_normalizado, log_monto, rango_horario, falta_bc")
 
# ==========================================================================
# 2.4 — SPLIT DE ÍNDICES TRAIN/TEST (ANTES de las features que usan
# estadísticas, para no filtrar información del test al entrenamiento)
# ==========================================================================
 
idx_train, idx_test = train_test_split(
    df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    stratify=df["fraude"]
)
df_train_stats = df.loc[idx_train]  # SOLO se usa para calcular estadísticas
 
# ==========================================================================
# 2.5 — FEATURE ENGINEERING (parte 2 — requiere estadísticas de train)
# ==========================================================================
 
# --- (e) Target encoding de 'j': tasa de fraude histórica de la categoría ---
# Calculado exclusivamente con df_train_stats. Las categorías de 'j' que
# aparecen en test pero no en train (o con pocos casos) caen al promedio
# global de train gracias al suavizado.
encoding_j, media_global_fraude = target_encoding_suavizado(
    df_train_stats, "j", target="fraude", suavizado=TARGET_ENCODING_SUAVIZADO
)
df["target_encoding_j"] = df["j"].map(encoding_j).fillna(media_global_fraude)
 
# --- (f) Desvío de precio respecto a la mediana de su categoría/producto ---
# Un precio muy por debajo de lo habitual para esa categoría/producto es un
# patrón clásico de fraude ("demasiado bueno para ser verdad"). Ratio, no
# diferencia absoluta, para que sea comparable entre categorías de distinto
# rango de precio. 0 = está justo en la mediana; -0.5 = cuesta la mitad de
# lo habitual; +1 = cuesta el doble.
mediana_monto_global = df_train_stats["monto"].median()
 
mediana_monto_j = df_train_stats.groupby("j")["monto"].median()
df["desvio_precio_categoria"] = (
    df["monto"] / df["j"].map(mediana_monto_j).fillna(mediana_monto_global) - 1
)

 
mediana_monto_i = df_train_stats.groupby("i")["monto"].median()
df["desvio_precio_producto"] = (
    df["monto"] / df["i"].map(mediana_monto_i).fillna(mediana_monto_global) - 1
)
 
print("Feature engineering (parte 2 — con estadísticas de train) aplicado: "
      "target_encoding_j, desvio_precio_categoria, desvio_precio_producto")
 
# ==========================================================================
# 3. PREPARAR VARIABLES
# ==========================================================================
 
X = df[FEATURES].copy()
y = df["fraude"].copy()
monto_completo = df["monto"].copy()  # se guarda aparte, se use o no como feature
 
cols_categoricas = [c for c in FEATURES if c in FEATURES_CATEGORICAS]
cols_numericas = [c for c in FEATURES if c not in FEATURES_CATEGORICAS]
 
if cols_categoricas:
    X[cols_categoricas] = X[cols_categoricas].fillna("__NA__")
    X = pd.get_dummies(X, columns=cols_categoricas, drop_first=True)
    print(f"Categóricas codificadas ({cols_categoricas}) -> "
          f"{X.shape[1]} columnas finales")
 
# Imputación de numéricas: la mediana se calcula SOLO con train, y se aplica
# igual a train y test (otra vez, para no filtrar información del test).
for col in cols_numericas:
    if X[col].isna().any():
        mediana_train = X.loc[idx_train, col].median()
        n_na = X[col].isna().sum()
        X[col] = X[col].fillna(mediana_train)
        print(f"'{col}': {n_na:,} NAs imputados con la mediana de train ({mediana_train:.2f})")
 
# ==========================================================================
# 4. SPLIT TRAIN / TEST (usando los mismos índices ya definidos arriba)
# ==========================================================================
 
X_train, X_test = X.loc[idx_train], X.loc[idx_test]
y_train, y_test = y.loc[idx_train], y.loc[idx_test]
monto_train, monto_test = monto_completo.loc[idx_train], monto_completo.loc[idx_test]
 
print(f"\nTrain: {len(X_train):,} filas | Test: {len(X_test):,} filas")
print(f"Fraude en train: {y_train.mean()*100:.2f}% | "
      f"Fraude en test: {y_test.mean()*100:.2f}%")
 
# ==========================================================================
# 5. ESCALAR
# ==========================================================================
 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
# ==========================================================================
# 6. ENTRENAR MODELO
# ==========================================================================
 
modelo = LogisticRegression(**MODEL_PARAMS)
modelo.fit(X_train_scaled, y_train)
 
print("\n--- Coeficientes del modelo ---")
for feat, coef in zip(X.columns, modelo.coef_[0]):
    signo = "+" if coef > 0 else ""
    print(f"  {feat}: {signo}{coef:.3f}")
 
# ==========================================================================
# 7. ELEGIR EL UMBRAL SEGÚN EL MÉTODO CONFIGURADO
# ==========================================================================
 
proba_test = modelo.predict_proba(X_test_scaled)[:, 1]
 
umbral_f1, umbrales_f1, f1_scores_por_umbral = buscar_umbral_max_f1(y_test, proba_test)
umbral_costo, umbrales_costo, costos_por_umbral = buscar_umbral_costo_minimo(
    y_test, proba_test, monto_test
)
umbral_fpr, fpr_logrado, tpr_logrado, fpr_curva, tpr_curva, _ = buscar_umbral_fpr_maximo(
    y_test, proba_test, FPR_MAXIMO
)
 
print(f"\nUmbral que maximiza F1:        {umbral_f1:.3f}  (F1={f1_scores_por_umbral.max():.3f})")
print(f"Umbral que minimiza el costo:  {umbral_costo:.3f}  (costo=${costos_por_umbral.min():,.2f})")
print(f"Umbral con FPR <= {FPR_MAXIMO*100:.1f}%:     {umbral_fpr:.3f}  "
      f"(FPR logrado={fpr_logrado*100:.2f}%, recall logrado={tpr_logrado*100:.2f}%)")
 
if METODO_UMBRAL == "f1":
    UMBRAL = umbral_f1
elif METODO_UMBRAL == "costo":
    UMBRAL = umbral_costo
elif METODO_UMBRAL == "fpr_maximo":
    UMBRAL = umbral_fpr
else:
    UMBRAL = UMBRAL_MANUAL
 
print(f"\n>>> Umbral usado para la evaluación final ({METODO_UMBRAL}): {UMBRAL:.3f}")
 
pred_test = (proba_test >= UMBRAL).astype(int)
 
# ==========================================================================
# 8. EVALUAR
# ==========================================================================
 
print(f"\n{'='*60}")
print(f"EVALUACIÓN — features={FEATURES}")
print(f"{'='*60}\n")
 
print(classification_report(y_test, pred_test, target_names=["No fraude", "Fraude"]))
 
auc = roc_auc_score(y_test, proba_test)
pr_auc = average_precision_score(y_test, proba_test)
prevalencia = y_test.mean()
print(f"ROC-AUC: {auc:.4f}  (0.5 = azar, 1.0 = perfecto)")
print(f"PR-AUC (Average Precision): {pr_auc:.4f}  (línea base de azar = {prevalencia:.4f} [{prevalencia*100:.2f}%], 1.0 = perfecto)")
 
cm = confusion_matrix(y_test, pred_test)
cm_pct = cm / cm.sum() * 100
 
costo_total, costo_fn, costo_fp, n_fn, n_fp = calcular_costo_total(
    y_test, proba_test, monto_test, UMBRAL
)
print(f"\n--- Costo de negocio con umbral={UMBRAL:.3f} ---")
print(f"  Falsos negativos: {n_fn:,}  -> costo (monto perdido): ${costo_fn:,.2f}")
print(f"  Falsos positivos: {n_fp:,}  -> costo (reputación + revisión): ${costo_fp:,.2f}")
print(f"  Costo total: ${costo_total:,.2f}")
 
# ==========================================================================
# 9. GRÁFICOS
# ==========================================================================
 
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
 
# 9.1 — Matriz de confusión con conteo Y porcentaje del total
etiquetas = ["No fraude", "Fraude"]
im = axes[0, 0].imshow(cm, cmap="Blues")
axes[0, 0].set_xticks([0, 1]); axes[0, 0].set_xticklabels(etiquetas)
axes[0, 0].set_yticks([0, 1]); axes[0, 0].set_yticklabels(etiquetas)
axes[0, 0].set_xlabel("Predicho"); axes[0, 0].set_ylabel("Real")
axes[0, 0].set_title(f"Matriz de confusión (umbral={UMBRAL:.3f})")
for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        axes[0, 0].text(j, i, f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)",
                         ha="center", va="center", color=color, fontsize=11)
 
# 9.2 — Curva ROC (con el presupuesto de FPR y el punto elegido marcados)
fpr, tpr, _ = roc_curve(y_test, proba_test)
axes[0, 1].plot(fpr, tpr, color="#457B9D", linewidth=2, label=f"AUC = {auc:.3f}")
axes[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[0, 1].axvline(FPR_MAXIMO, color="orange", linestyle="--",
                    label=f"FPR máximo permitido = {FPR_MAXIMO*100:.0f}%")
axes[0, 1].scatter([fpr_logrado], [tpr_logrado], color="red", zorder=5,
                    label=f"Umbral FPR máx. (recall={tpr_logrado*100:.1f}%)")
axes[0, 1].set_xlabel("Tasa de falsos positivos")
axes[0, 1].set_ylabel("Tasa de verdaderos positivos")
axes[0, 1].set_title("Curva ROC")
axes[0, 1].legend(fontsize=8)

# 9.3 — Curva Precision-Recall (PR)
precisiones, recalls, _ = precision_recall_curve(y_test, proba_test)
prec_umbral = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0.0
rec_umbral = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0.0

axes[0, 2].plot(recalls, precisiones, color="#2A9D8F", linewidth=2, label=f"PR-AUC = {pr_auc:.3f}")
axes[0, 2].axhline(prevalencia, color="gray", linestyle="--", label=f"Azar = {prevalencia*100:.1f}%")
axes[0, 2].scatter([rec_umbral], [prec_umbral], color="red", zorder=5,
                    label=f"Umbral {UMBRAL:.3f} (P={prec_umbral*100:.1f}%, R={rec_umbral*100:.1f}%)")
axes[0, 2].set_xlabel("Recall (Sensibilidad)")
axes[0, 2].set_ylabel("Precision")
axes[0, 2].set_title("Curva Precision-Recall (PR)")
axes[0, 2].set_xlim([0.0, 1.0])
axes[0, 2].set_ylim([0.0, 1.05])
axes[0, 2].legend(fontsize=8)
 
# 9.4 — F1 vs umbral
axes[1, 0].plot(umbrales_f1, f1_scores_por_umbral, color="#2A9D8F")
axes[1, 0].axvline(umbral_f1, color="gray", linestyle="--",
                    label=f"Óptimo F1 = {umbral_f1:.3f}")
axes[1, 0].axvline(UMBRAL, color="red", linestyle=":", label=f"Umbral usado = {UMBRAL:.3f}")
axes[1, 0].set_xlabel("Umbral"); axes[1, 0].set_ylabel("F1-score")
axes[1, 0].set_title("F1-score según el umbral")
axes[1, 0].legend()
 
# 9.5 — Costo de negocio vs umbral
axes[1, 1].plot(umbrales_costo, costos_por_umbral, color="#E63946")
axes[1, 1].axvline(umbral_costo, color="gray", linestyle="--",
                    label=f"Óptimo costo = {umbral_costo:.3f}")
axes[1, 1].axvline(UMBRAL, color="blue", linestyle=":", label=f"Umbral usado = {UMBRAL:.3f}")
axes[1, 1].set_xlabel("Umbral"); axes[1, 1].set_ylabel("Costo total ($)")
axes[1, 1].set_title("Costo de negocio según el umbral")
axes[1, 1].legend()

# 9.6 — Precision y Recall según el umbral
prec_por_umbral = []
rec_por_umbral = []
for u in umbrales_f1:
    p = (proba_test >= u).astype(int)
    tp = ((y_test == 1) & (p == 1)).sum()
    fp = ((y_test == 0) & (p == 1)).sum()
    fn = ((y_test == 1) & (p == 0)).sum()
    prec_por_umbral.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
    rec_por_umbral.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

axes[1, 2].plot(umbrales_f1, prec_por_umbral, color="#1D3557", linewidth=1.8, label="Precision")
axes[1, 2].plot(umbrales_f1, rec_por_umbral, color="#E63946", linewidth=1.8, label="Recall")
axes[1, 2].axvline(UMBRAL, color="purple", linestyle=":", label=f"Umbral usado = {UMBRAL:.3f}")
axes[1, 2].set_xlabel("Umbral"); axes[1, 2].set_ylabel("Métrica (0 - 1)")
axes[1, 2].set_title("Precision y Recall según el umbral")
axes[1, 2].legend(fontsize=8)
 
plt.tight_layout()
plt.savefig(RUTA_SALIDA_GRAFICOS, dpi=120)
print(f"\nGráficos guardados en {RUTA_SALIDA_GRAFICOS}")
plt.show()
 