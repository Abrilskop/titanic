import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_JSON = os.path.join(BASE_DIR, "model.json")
PLOT_PNG = os.path.join(BASE_DIR, "pets_clasificador.png")

IMG_SIZE = 64
CELL = 8
BLOCK = 2
NBINS = 9
FEATURES_COUNT = ((IMG_SIZE // CELL) - BLOCK + 1) ** 2 * BLOCK * BLOCK * NBINS
ALPHA = 0.01

CLASES = {
    "cats": {"label": 0, "nombre": "Gato"},
    "dogs": {"label": 1, "nombre": "Perro"},
}


def es_imagen_valida(ruta):
    try:
        if ruta.lower().endswith(".db"):
            return False
        with Image.open(ruta) as im:
            im.load()
        return True
    except Exception:
        return False


def recortar_centro(imagen):
    w, h = imagen.size
    lado = min(w, h)
    x0 = (w - lado) // 2
    y0 = (h - lado) // 2
    return imagen.crop((x0, y0, x0 + lado, y0 + lado))


def ecualizar_gris(imagen):
    img = np.asarray(imagen, dtype=np.uint8)
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0].min()
    total = cdf[-1]
    if total - cdf_min == 0:
        return np.zeros_like(img)
    lut = (cdf - cdf_min) / (total - cdf_min) * 255.0
    lut = np.clip(lut, 0, 255)
    return lut[img]


def cargar_imagen_cuadrada(ruta):
    with Image.open(ruta) as im:
        im = im.convert("L")
        im = recortar_centro(im)
        im = im.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = ecualizar_gris(im).astype(np.float64) / 255.0
    return arr


def gradientes(img):
    n = img.shape[0]
    gx = np.empty_like(img)
    gy = np.empty_like(img)
    gx[:, 0] = img[:, 1] - img[:, 0]
    gx[:, -1] = img[:, -1] - img[:, -2]
    gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
    gy[0, :] = img[1, :] - img[0, :]
    gy[-1, :] = img[-1, :] - img[-2, :]
    gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0
    return gx, gy


def hist_hog(imagen):
    gx, gy = gradientes(imagen)
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0
    ncells = IMG_SIZE // CELL
    bin_idx = (ang // (180.0 / NBINS)).astype(np.int64) % NBINS
    row_cell = (np.arange(IMG_SIZE) // CELL)[:, None]
    col_cell = (np.arange(IMG_SIZE) // CELL)[None, :]
    flat_idx = (row_cell * ncells + col_cell) * NBINS + bin_idx
    hist = np.zeros(ncells * ncells * NBINS, dtype=np.float64)
    np.add.at(hist, flat_idx.ravel(), mag.ravel())
    return hist.reshape(ncells, ncells, NBINS)


def bloque_hog(hist):
    ncells = IMG_SIZE // CELL
    partes = []
    for by in range(ncells - BLOCK + 1):
        for bx in range(ncells - BLOCK + 1):
            v = hist[by:by + BLOCK, bx:bx + BLOCK].ravel().copy()
            v = v / (np.sqrt(np.sum(v ** 2)) + 1e-5)
            v = np.clip(v, 0.0, 0.2)
            v = v / (np.sqrt(np.sum(v ** 2)) + 1e-5)
            partes.append(v)
    return np.concatenate(partes)


def extraer_hog(imagen):
    return bloque_hog(hist_hog(imagen))


# ============================================================
# SECCION 1: LEER EL DATASET PERROS Y GATOS
# ============================================================
print("=" * 60)
print("SECCION 1: LEER EL DATASET PERROS Y GATOS")
print("=" * 60)

rows = []
corruptas = {}

for carpeta, info in CLASES.items():
    directorio = os.path.join(DATA_DIR, carpeta)
    archivos = sorted(os.listdir(directorio))
    validos = [a for a in archivos if es_imagen_valida(os.path.join(directorio, a))]
    corruptas[info["nombre"]] = len(archivos) - len(validos)
    for a in validos:
        rows.append({"ruta": os.path.join(directorio, a), "label": info["label"],
                     "clase": info["nombre"]})

df = pd.DataFrame(rows)
print(f"\nImagenes corruptas descartadas: {corruptas}")
print(f"\nDimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
print("\nPrimeras 5 filas:")
print(df.head().to_string(index=False))
print(f"\nDistribucion de clases:\n{df['clase'].value_counts().to_string()}")

# ============================================================
# SECCION 2: EXPLORAR LOS DATOS (DESCRIBE)
# ============================================================
print("\n" + "=" * 60)
print("SECCION 2: EXPLORAR LOS DATOS (DESCRIBE)")
print("=" * 60)

def dimensiones_imagen(fila):
    with Image.open(fila["ruta"]) as im:
        w, h = im.size
        return pd.Series({"ancho": w, "alto": h, "aspecto": round(w / h, 3)})

print("Calculando dimensiones de las imagenes...")
dim = df.apply(dimensiones_imagen, axis=1)
df_out = pd.concat([df, dim], axis=1)

print("\n--- Dimensiones de las imagenes originales ---")
print(df_out[["ancho", "alto", "aspecto"]].describe().round(2).to_string())

print("\n--- Estadisticas de pixeles (recorte centrado, ecualizado, gris 64x64) ---")
muestras = df.sample(n=min(400, len(df)), random_state=7)
pixeles = np.stack([cargar_imagen_cuadrada(r) for r in muestras["ruta"]]).reshape(-1, IMG_SIZE * IMG_SIZE)
stats = pd.DataFrame({
    "media_pixel": pixeles.mean(axis=1),
    "desv_pixel": pixeles.std(axis=1),
    "min_pixel": pixeles.min(axis=1),
    "max_pixel": pixeles.max(axis=1),
})
print(stats.describe().round(3).to_string())

print(f"\nValores nulos por columna:\n{df_out.isnull().sum().to_string()}")

# ============================================================
# SECCION 3: MODELAR CON REGRESION LOGISTICA + METRICAS
# ============================================================
print("\n" + "=" * 60)
print("SECCION 3: REGRESION LOGISTICA + METRICAS DE CLASIFICACION")
print("=" * 60)

print("\nExtrayendo descriptores HOG (histograma de gradientes orientados)...")
X = np.stack([extraer_hog(cargar_imagen_cuadrada(r)) for r in df["ruta"]])
y = df["label"].to_numpy()

print(f"Matriz de features: {X.shape[0]} muestras x {X.shape[1]} caracteristicas")
print("\n--- Descriptores utilizados (feature engineering) ---")
print(f"  {NBINS} histogramas de orientacion de gradientes por celda de {CELL}x{CELL}")
print(f"  Bloques de normalizacion de {BLOCK}x{BLOCK} celdas con norma L2-Hys")
print(f"  Preprocesado: escala de grises, recorte central cuadrado, ecualizacion de histograma")
print(f"  Total: {FEATURES_COUNT} caracteristicas")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SGDClassifier(loss="log_loss", max_iter=5000, tol=1e-5,
                      random_state=42, alpha=ALPHA)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n--- Metricas de Clasificacion Binaria ---")
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision : {precision:.4f}  ({precision*100:.2f}%)")
print(f"  Recall    : {recall:.4f}  ({recall*100:.2f}%)")
print(f"  F1-Score  : {f1:.4f}")

print(f"\n--- Reporte de Clasificacion ---")
print(classification_report(y_test, y_pred, target_names=["Gato", "Perro"]))

print(f"\n--- Matriz de Confusion ---")
cm = confusion_matrix(y_test, y_pred)
print(f"                 [VN: {cm[0][0]}  FP: {cm[0][1]}]")
print(f"                 [FN: {cm[1][0]}  VP: {cm[1][1]}]")

print(f"\n--- Regiones y orientaciones mas relevantes ---")
ncells = IMG_SIZE // CELL
peso_bloques = np.abs(model.coef_[0].reshape(
    ncells - BLOCK + 1, ncells - BLOCK + 1, BLOCK, BLOCK, NBINS))
peso_celdas = np.zeros((ncells, ncells))
for by in range(ncells - BLOCK + 1):
    for bx in range(ncells - BLOCK + 1):
        peso_celdas[by:by + BLOCK, bx:bx + BLOCK] += peso_bloques[by, bx].sum(axis=-1)
importancia_celda = np.unravel_index(np.argmax(peso_celdas), peso_celdas.shape)
oriente = np.abs(model.coef_[0].reshape(-1, NBINS)).sum(axis=0)
print(f"  Celda mas relevante del mapa HOG: (fila={importancia_celda[0]}, col={importancia_celda[1]})")
print(f"  Peso por orientacion (grados): " + ", ".join(
    f"{int(round(i*180/NBINS))}={oriente[i]:.1f}" for i in range(NBINS)))

# ============================================================
# SECCION 4: GRAFICAS
# ============================================================
print("\n" + "=" * 60)
print("SECCION 4: GENERANDO GRAFICAS...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Perros vs Gatos - Regresion Logistica + HOG - Clasificacion Binaria",
             fontsize=16, fontweight="bold")

x_sig = np.linspace(-10, 10, 200)
y_sig = 1 / (1 + np.exp(-x_sig))

ax1 = axes[0, 0]
ax1.plot(x_sig, y_sig, color="blue", linewidth=2)
ax1.axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="Limite = 0.5")
ax1.axvline(x=0, color="gray", linestyle=":", linewidth=1)
ax1.set_title("Funcion Sigmoide: y = 1/(1 + e^(-x))")
ax1.set_xlabel("x (suma ponderada de descriptores HOG)")
ax1.set_ylabel("y (probabilidad de ser perro)")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2,
            xticklabels=["Gato", "Perro"], yticklabels=["Gato", "Perro"])
ax2.set_title("Matriz de Confusion")
ax2.set_xlabel("Prediccion")
ax2.set_ylabel("Real")

ax3 = axes[1, 0]
im = ax3.imshow(peso_celdas, cmap="magma", interpolation="bilinear")
plt.colorbar(im, ax=ax3, shrink=0.8, label="Peso acumulado por celda")
for i in range(peso_celdas.shape[0]):
    ax3.axhline(i - 0.5, color="white", linewidth=0.4)
    ax3.axvline(i - 0.5, color="white", linewidth=0.4)
ax3.set_title("Zonas de la imagen mas relevantes (celdas HOG)")
ax3.set_xlabel("Columna de celda")
ax3.set_ylabel("Fila de celda")
ax3.grid(False)

ax4 = axes[1, 1]
ax4.plot(x_sig, y_sig, color="blue", linewidth=2)
ax4.axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="Limite = 0.5")
probs_test = model.predict_proba(X_test_scaled)[:, 1]
ejemplos = np.random.RandomState(0).choice(len(y_test), size=6, replace=False)
for i in ejemplos:
    p = probs_test[i]
    prob_guardada = min(max(p, 1e-6), 1 - 1e-6)
    z = float(np.log(prob_guardada / (1 - prob_guardada)))
    color = "green" if p >= 0.5 else "red"
    nombre = "Perro" if y_test[i] == 1 else "Gato"
    ax4.scatter(z, p, color=color, s=70, zorder=5)
    ax4.annotate(nombre, (z, p), textcoords="offset points", xytext=(7, 7), fontsize=8)
ax4.set_title("Prediccion de ejemplos (puntos sobre la sigmoide)")
ax4.set_xlabel("x (suma ponderada)")
ax4.set_ylabel("y (probabilidad de ser perro)")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_PNG, dpi=150, bbox_inches="tight")
print(f"Graficas guardadas en: {PLOT_PNG}")

# ============================================================
# SECCION 5: EXPORTAR MODELO PARA EL SERVIDOR WEB
# ============================================================
print("\n" + "=" * 60)
print("SECCION 5: EXPORTAR MODELO A model.json")
print("=" * 60)

payload = {
    "modelo": "regresion_logistica_hog",
    "imagen_size": IMG_SIZE,
    "hog": {"cell": CELL, "block": BLOCK, "bins": NBINS},
    "clases": {"0": "Gato", "1": "Perro"},
    "label_1": "Perro",
    "coef": [float(c) for c in model.coef_[0]],
    "intercept": float(model.intercept_[0]),
    "scaler_mean": [float(m) for m in scaler.mean_],
    "scaler_scale": [float(s) for s in scaler.scale_],
}

with open(MODEL_JSON, "w", encoding="utf-8") as f:
    json.dump(payload, f)

print(f"Modelo exportado a: {MODEL_JSON}")
print(f"  - {len(payload['coef'])} coeficientes (descriptores HOG)")
print(f"  - intercept = {payload['intercept']:+.6f}")
print(f"  - accuracy = {accuracy:.4f}")

print("\nPrograma finalizado.")