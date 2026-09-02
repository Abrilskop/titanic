import sharp from "sharp";

export function buildModel(payload) {
  return {
    imagenSize: payload.imagen_size,
    cell: payload.hog.cell,
    block: payload.hog.block,
    bins: payload.hog.bins,
    coef: Float64Array.from(payload.coef),
    intercept: payload.intercept,
    mean: Float64Array.from(payload.scaler_mean),
    scale: Float64Array.from(payload.scaler_scale),
    modelo: payload.modelo,
    clases: payload.clases || { 0: "Gato", 1: "Perro" },
  };
}

function mod180(deg) {
  return ((deg % 180) + 180) % 180;
}

export function ecualizar(data) {
  const hist = new Float64Array(256);
  for (let i = 0; i < data.length; i++) {
    hist[data[i]] += 1;
  }
  const cdf = new Float64Array(256);
  let acc = 0;
  for (let i = 0; i < 256; i++) {
    acc += hist[i];
    cdf[i] = acc;
  }
  let cdfMin = 0;
  for (let i = 0; i < 256; i++) {
    if (cdf[i] > 0) {
      cdfMin = cdf[i];
      break;
    }
  }
  const total = cdf[255];
  const out = new Float64Array(data.length);
  if (total - cdfMin === 0) {
    return out;
  }
  for (let i = 0; i < data.length; i++) {
    out[i] = ((cdf[data[i]] - cdfMin) / (total - cdfMin)) * 255.0;
  }
  return out;
}

function gradientes(img, n) {
  const at = (y, x) => img[y * n + x];
  const gx = new Float64Array(n * n);
  const gy = new Float64Array(n * n);
  for (let y = 0; y < n; y++) {
    gx[y * n] = at(y, 1) - at(y, 0);
    gx[y * n + n - 1] = at(y, n - 1) - at(y, n - 2);
    for (let x = 1; x < n - 1; x++) {
      gx[y * n + x] = (at(y, x + 1) - at(y, x - 1)) / 2.0;
    }
  }
  for (let x = 0; x < n; x++) {
    gy[x] = at(1, x) - at(0, x);
    gy[(n - 1) * n + x] = at(n - 1, x) - at(n - 2, x);
  }
  for (let y = 1; y < n - 1; y++) {
    for (let x = 0; x < n; x++) {
      gy[y * n + x] = (at(y + 1, x) - at(y - 1, x)) / 2.0;
    }
  }
  return { gx, gy };
}

export function extraerHog(img, model) {
  const n = model.imagenSize;
  const { cell, block, bins } = model;
  const ncells = Math.floor(n / cell);
  const binWidth = 180.0 / bins;
  const { gx, gy } = gradientes(img, n);
  const hist = new Float64Array(ncells * ncells * bins);
  for (let y = 0; y < n; y++) {
    const rowCell = Math.floor(y / cell);
    for (let x = 0; x < n; x++) {
      const colCell = Math.floor(x / cell);
      const mag = Math.hypot(gx[y * n + x], gy[y * n + x]);
      const ang = mod180((Math.atan2(gy[y * n + x], gx[y * n + x]) * 180.0) / Math.PI);
      const binIdx = Math.floor(ang / binWidth) % bins;
      hist[(rowCell * ncells + colCell) * bins + binIdx] += mag;
    }
  }
  const features = new Float64Array(
    (ncells - block + 1) * (ncells - block + 1) * block * block * bins
  );
  let pos = 0;
  for (let by = 0; by < ncells - block + 1; by++) {
    for (let bx = 0; bx < ncells - block + 1; bx++) {
      const v = new Float64Array(block * block * bins);
      let k = 0;
      for (let dy = 0; dy < block; dy++) {
        for (let dx = 0; dx < block; dx++) {
          const base = ((by + dy) * ncells + (bx + dx)) * bins;
          for (let b = 0; b < bins; b++) {
            v[k++] = hist[base + b];
          }
        }
      }
      let sq = 0;
      for (let i = 0; i < v.length; i++) sq += v[i] * v[i];
      const n1 = Math.sqrt(sq) + 1e-5;
      for (let i = 0; i < v.length; i++) v[i] = v[i] / n1;
      for (let i = 0; i < v.length; i++) if (v[i] > 0.2) v[i] = 0.2;
      sq = 0;
      for (let i = 0; i < v.length; i++) sq += v[i] * v[i];
      const n2 = Math.sqrt(sq) + 1e-5;
      for (let i = 0; i < v.length; i++) v[i] = v[i] / n2;
      features.set(v, pos);
      pos += v.length;
    }
  }
  return features;
}

export function predecirFeatureVector(features, model) {
  let z = model.intercept;
  for (let i = 0; i < model.coef.length; i++) {
    z += model.coef[i] * ((features[i] - model.mean[i]) / model.scale[i]);
  }
  const probPerro = 1 / (1 + Math.exp(-z));
  const clasificacion = probPerro >= 0.5 ? "Perro" : "Gato";
  const probClase = probPerro >= 0.5 ? probPerro : 1 - probPerro;
  return { clase: clasificacion, etiqueta: clasificacion, probPerro, probClase, z };
}

export async function prepararPixelesGrises(buffer, model) {
  const meta = await sharp(buffer).metadata();
  const { width: w, height: h } = meta;
  const lado = Math.min(w, h);
  const left = Math.floor((w - lado) / 2);
  const top = Math.floor((h - lado) / 2);
  const raw = await sharp(buffer)
    .grayscale()
    .extract({ left, top, width: lado, height: lado })
    .resize(model.imagenSize, model.imagenSize)
    .raw()
    .toBuffer({ resolveWithObject: true });
  if (raw.info.channels !== 1) {
    throw new Error("No se pudo convertir la imagen a gris.");
  }
  const pixeles = ecualizar(raw.data);
  return { pixeles, ancho: w, alto: h };
}

export async function clasificarBuffer(buffer, model) {
  const { pixeles, ancho, alto } = await prepararPixelesGrises(buffer, model);
  const features = extraerHog(pixeles, model);
  const resultado = predecirFeatureVector(features, model);
  return { ...resultado, features, imagen: { ancho, alto } };
}