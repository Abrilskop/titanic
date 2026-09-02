# Clasificador Perros vs Gatos 🐶🐱

Clasificación binaria de imágenes (gato o perro) con **regresión logística**
sobre descriptores **HOG** (histograma de gradientes orientados), entrenado con
el dataset [Dogs vs. Cats](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
de Microsoft/Kaggle (~25.000 imágenes).

El proyecto replica la estructura del script de Titanic, pero agregando un
**frontend web moderno** y un servidor Node que usa el mismo modelo para
clasificar imágenes en vivo.

## Resultados

| Métrica   | Valor  |
| --------- | ------ |
| Accuracy  | 72.44% |
| Precision | 71.74% |
| Recall    | 74.04% |
| F1-Score  | 0.7287 |

## Estructura

```
pets-classifier/
├── train_lr.py          # Entrenamiento en 5 secciones (leer, explorar,
│                        #   modelar, graficar, exportar model.json)
├── model.json           # Pesos de la regresión logística + escalador
├── pets_clasificador.png# Gráficas generadas (sigmoide, matriz de confusión...)
├── data/                # Imágenes (ignoradas por git, ~3 GB sin comprimir)
├── server/              # API Express (Node) con el port de la pipeline
│   ├── index.js         #   Endpoints y servido estático del frontend
│   └── pipeline.js      #   Preprocesado + HOG + logística, en JS puro
└── frontend/            # UI web (HTML/CSS/JS sin frameworks)
    ├── index.html
    ├── styles.css
    └── app.js
```

## Cómo funciona

1. **Preprocesado** (idéntico en Python y JS para garantizar consistencia):
   escala de grises → recorte central cuadrado → ecualización de histograma
   (CDF) → redimensionado a 64×64.
2. **Features HOG**: 9 histogramas de orientación de gradientes por celda de
   8×8, normalización L2-Hys por bloques de 2×2 celdas → **1764 descriptores**.
3. **Regresión logística**: `z = coef · ((x - media) / desviación) + b`,
   `p = 1 / (1 + e⁻ᶻ)`. La clase 1 (positivo) es *Perro*.

## Puesta en marcha

### 1. Datos (opcional, si vas a reentrenar)

El dataset se descarga desde el [Download Center de Microsoft](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
(`kagglecatsanddogs_5340.zip`). Se espera la siguiente estructura:

```
pets-classifier/data/
├── cats/   # ~12.500 JPG de gatos
└── dogs/   # ~12.500 JPG de perros
```

### 2. Entrenar (genera `model.json` y las gráficas)

```bash
pip install numpy pandas matplotlib seaborn pillow scikit-learn
python train_lr.py
```

Un modelo ya entrenado viene versionado en `model.json`, así que este paso es
opcional si solo querés correr la web.

### 3. Levantar el servidor

```bash
cd server
npm install
npm start
# → Clasificador listo en http://localhost:3000
```

### 4. Usar

Abrí `http://localhost:3000`, arrastrá una foto de un gato o un perro (o pegá
una con Ctrl+V, o tocá la zona para elegir archivo) y mirá el resultado:
emoji, probabilidad animada y, si el modelo está muy seguro, confeti 🎉.

También podés probar la API directamente:

```bash
curl -X POST http://localhost:3000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"image\":\"<base64 de la imagen>\"}"
```

## Notas

- El dataset oficial trae ~1.738 imágenes corruptas; `train_lr.py` las descarta.
- El preprocesado/HOG se implementó a mano (espejo Python ↔ JS) para poder
  replicar exactamente la matemática del modelo en el navegador sin librerías
  de ML en Node.
- Hay una pequeña deriva entre el entrenamiento (pillow) y la inferencia
  (sharp) por diferencias de redimensionado; es despreciable en la práctica
  (dif. media de probabilidad ~3.4%).