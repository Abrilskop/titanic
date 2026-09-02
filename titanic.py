import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk, messagebox

# ============================================================
# SECCION 1: LEER EL DATASET TITANIC
# ============================================================
print("=" * 60)
print("SECCION 1: LEER EL DATASET TITANIC")
print("=" * 60)

df = sns.load_dataset("titanic")
print(f"\nDimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"\nPrimeras 5 filas:")
print(df.head())
print(f"\nColumnas disponibles:\n{list(df.columns)}")

# ============================================================
# SECCION 2: EXPLORAR LOS DATOS (DESCRIBE)
# ============================================================
print("\n" + "=" * 60)
print("SECCION 2: EXPLORAR LOS DATOS (DESCRIBE)")
print("=" * 60)

print(f"\nEstadisticas generales:")
print(df.describe())

print(f"\nValores nulos por columna:")
print(df.isnull().sum())

print(f"\nTipos de dato:")
print(df.dtypes)

# --- Seleccion de descriptores candidatos ---
# Se excluyen: passengerid, name, ticket, cabin (claves/irrelevantes)
# Se usan: pclass, sex, age, sibsp, parch, fare, embarked
print("\n--- Descriptores seleccionados como candidatos ---")
print("  pclass  -> Clase del pasajero (1ra, 2da, 3ra)")
print("  sex     -> Sexo (male/female)")
print("  age     -> Edad")
print("  sibsp   -> Hermanos/conyuges a bordo")
print("  parch   -> Padres/hijos a bordo")
print("  fare    -> Tarifa pagada")
print("  embarked-> Puerto de embarque (S/C/Q)")

# --- Preprocesamiento ---
df_model = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()

# Rellenar nulos
df_model["age"] = df_model["age"].fillna(df_model["age"].median())
df_model["fare"] = df_model["fare"].fillna(df_model["fare"].median())
df_model["embarked"] = df_model["embarked"].fillna(df_model["embarked"].mode()[0])

# Encoding: sex -> 0/1
df_model["sex"] = df_model["sex"].map({"male": 0, "female": 1})

# Encoding: embarked -> one-hot
df_model = pd.get_dummies(df_model, columns=["embarked"], drop_first=True)

print(f"\nDataset preprocesado ({df_model.shape[0]} filas x {df_model.shape[1]} columnas):")
print(df_model.head())

# ============================================================
# SECCION 3: MODELAR CON REGRESION LOGISTICA + METRICAS
# ============================================================
print("\n" + "=" * 60)
print("SECCION 3: REGRESION LOGISTICA + METRICAS DE CLASIFICACION")
print("=" * 60)

X = df_model.drop("survived", axis=1)
y = df_model["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Metricas de clasificacion
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
print(classification_report(y_test, y_pred, target_names=["No Sobrevivio", "Sobrevivio"]))

print(f"\n--- Matriz de Confusion ---")
cm = confusion_matrix(y_test, y_pred)
print(f"  No Sobrevivio | Sobrevivio")
print(f"  [VN: {cm[0][0]}  FP: {cm[0][1]}]")
print(f"  [FN: {cm[1][0]}  VP: {cm[1][1]}]")

print(f"\n--- Coeficientes del modelo ---")
for name, coef in zip(X.columns, model.coef_[0]):
    print(f"  {name:>15s}: {coef:+.4f}")
print(f"  {'Intercept':>15s}: {model.intercept_[0]:+.4f}")

# ============================================================
# SECCION 4: GRAFICAS
# ============================================================
print("\n" + "=" * 60)
print("SECCION 4: GENERANDO GRAFICAS...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Titanic - Regresion Logistica - Clasificacion Binaria", fontsize=16, fontweight="bold")

# --- Grafica 1: Funcion Sigmoide ---
ax1 = axes[0, 0]
x_sig = np.linspace(-10, 10, 200)
y_sig = 1 / (1 + np.exp(-x_sig))
ax1.plot(x_sig, y_sig, color="blue", linewidth=2)
ax1.axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="Limite = 0.5")
ax1.axvline(x=0, color="gray", linestyle=":", linewidth=1)
ax1.set_title("Funcion Sigmoide: y = 1/(1 + e^(-x))")
ax1.set_xlabel("x")
ax1.set_ylabel("y (probabilidad)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Grafica 2: Matriz de Confusion ---
ax2 = axes[0, 1]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2,
            xticklabels=["No Sobrevivio", "Sobrevivio"],
            yticklabels=["No Sobrevivio", "Sobrevivio"])
ax2.set_title("Matriz de Confusion")
ax2.set_xlabel("Prediccion")
ax2.set_ylabel("Real")

# --- Grafica 3: Importancia de features (coeficientes) ---
ax3 = axes[1, 0]
feature_names = X.columns
coefs = model.coef_[0]
colors = ["green" if c > 0 else "red" for c in coefs]
ax3.barh(feature_names, coefs, color=colors)
ax3.set_title("Coeficientes del Modelo (importancia)")
ax3.set_xlabel("Coeficiente")
ax3.axvline(x=0, color="black", linewidth=0.8)
ax3.grid(True, alpha=0.3, axis="x")

# --- Grafica 4: Prediccion de supervivencia (punto sobre la sigmoide) ---
ax4 = axes[1, 1]
ax4.plot(x_sig, y_sig, color="blue", linewidth=2)
ax4.axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="Limite = 0.5")
# Ejemplo de varios pasajeros
pasajeros = pd.DataFrame([
    {"nombre": "Pasajero A (Hombre 3ra)", "row": {"pclass":3,"sex":0,"age":25,"sibsp":0,"parch":0,"fare":7,"embarked_Q":0,"embarked_S":1}},
    {"nombre": "Pasajero B (Mujer 1ra)", "row": {"pclass":1,"sex":1,"age":29,"sibsp":1,"parch":0,"fare":100,"embarked_Q":0,"embarked_S":1}},
    {"nombre": "Pasajero C (Mujer 3ra)", "row": {"pclass":3,"sex":1,"age":26,"sibsp":0,"parch":0,"fare":8,"embarked_Q":0,"embarked_S":1}},
])
for _, p in pasajeros.iterrows():
    inp = pd.DataFrame([p["row"]])
    prob = model.predict_proba(scaler.transform(inp))[0][1]
    z = np.log(prob / (1 - prob)) if (0 < prob < 1) else (10 if prob == 1 else -10)
    color = "green" if prob >= 0.5 else "red"
    ax4.scatter(z, prob, color=color, s=60, zorder=5)
    ax4.annotate(p["nombre"], (z, prob), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax4.set_title("Prediccion de supervivencia (puntos en la sigmoide)")
ax4.set_xlabel("x (suma ponderada)")
ax4.set_ylabel("y (probabilidad de sobrevivir)")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("titanic_graficas.png", dpi=150, bbox_inches="tight")
print("Graficas guardadas en: titanic_graficas.png")
plt.show()

# ============================================================
# SECCION 5: INTERFAZ GRAFICA PARA CLASIFICACION BINARIA
# ============================================================
print("\n" + "=" * 60)
print("SECCION 5: INTERFAZ GRAFICA - CLASIFICACION BINARIA")
print("=" * 60)
print("Abriendo ventana grafica... (cerrar ventana para finalizar)")

root = tk.Tk()
root.title("Titanic - Clasificacion Binaria - Sobrevivio o No?")
root.geometry("500x520")
root.resizable(False, False)
root.configure(bg="#2c3e50")

title_label = tk.Label(root, text="PREDICCION DE SUPERVIVENCIA TITANIC",
                       font=("Arial", 14, "bold"), bg="#2c3e50", fg="white")
title_label.pack(pady=10)

frame = tk.Frame(root, bg="#34495e", padx=20, pady=15)
frame.pack(padx=20, pady=5, fill="x")

labels_fields = [
    ("Clase (1, 2, 3):", "entry_pclass"),
    ("Sexo (0=Hombre, 1=Mujer):", "entry_sex"),
    ("Edad:", "entry_age"),
    ("Hermanos/Conyuges:", "entry_sibsp"),
    ("Padres/Hijos:", "entry_parch"),
    ("Tarifa:", "entry_fare"),
    ("Puerto (0=C, 1=Q, 2=S):", "entry_embarked"),
]

entries = {}
for i, (text, name) in enumerate(labels_fields):
    lbl = tk.Label(frame, text=text, font=("Arial", 11), bg="#34495e", fg="white", anchor="w")
    lbl.grid(row=i, column=0, sticky="w", pady=5)
    ent = tk.Entry(frame, font=("Arial", 11), width=20)
    ent.grid(row=i, column=1, pady=5, padx=(10, 0))
    entries[name] = ent

# Valores por defecto
entries["entry_pclass"].insert(0, "3")
entries["entry_sex"].insert(0, "0")
entries["entry_age"].insert(0, "25")
entries["entry_sibsp"].insert(0, "0")
entries["entry_parch"].insert(0, "0")
entries["entry_fare"].insert(0, "7.25")
entries["entry_embarked"].insert(0, "2")

result_frame = tk.Frame(root, bg="#2c3e50")
result_frame.pack(pady=10)

result_label = tk.Label(result_frame, text="", font=("Arial", 16, "bold"),
                        bg="#2c3e50", fg="white", width=35)
result_label.pack()


def predecir():
    try:
        pclass = float(entries["entry_pclass"].get())
        sex = float(entries["entry_sex"].get())
        age = float(entries["entry_age"].get())
        sibsp = float(entries["entry_sibsp"].get())
        parch = float(entries["entry_parch"].get())
        fare = float(entries["entry_fare"].get())
        embarked = float(entries["entry_embarked"].get())

        input_data = pd.DataFrame([{
            "pclass": pclass, "sex": sex, "age": age,
            "sibsp": sibsp, "parch": parch, "fare": fare,
            "embarked_Q": 1 if embarked == 1 else 0,
            "embarked_S": 1 if embarked == 2 else 0,
        }])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        if prediction == 1:
            result_label.config(text=f"SOBREVIVIO  ({probability[1]*100:.1f}%)", fg="#2ecc71")
        else:
            result_label.config(text=f"NO SOBREVIVIO  ({probability[0]*100:.1f}%)", fg="#e74c3c")

    except ValueError:
        messagebox.showerror("Error", "Ingrese valores numericos validos en todos los campos.")


btn_predict = tk.Button(root, text="PREDECIR", font=("Arial", 13, "bold"),
                        bg="#27ae60", fg="white", width=20, height=2,
                        command=predecir, cursor="hand2")
btn_predict.pack(pady=10)

root.mainloop()

print("\nPrograma finalizado.")
