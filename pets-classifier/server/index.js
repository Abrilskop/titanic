import express from "express";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildModel, clasificarBuffer } from "./pipeline.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, "..");
const MODEL_PATH = path.join(ROOT, "model.json");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: "15mb" }));

const model = buildModel(JSON.parse(await readFile(MODEL_PATH, "utf-8")));

app.use(express.static(path.join(ROOT, "frontend")));

app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    modelo: model.modelo,
    features: model.coef.length,
    accesos: 1,
  });
});

app.post("/api/predict", async (req, res) => {
  const inicio = Date.now();
  try {
    const imagen = (req.body && req.body.image) || "";
    if (!imagen) {
      return res.status(400).json({ ok: false, error: "No se recibio ninguna imagen." });
    }
    const base64 = imagen.includes(",") ? imagen.split(",")[1] : imagen;
    const buffer = Buffer.from(base64, "base64");

    const resultado = await clasificarBuffer(buffer, model);
    resultado.ok = true;
    resultado.ms = Date.now() - inicio;
    return res.json(resultado);
  } catch (err) {
    return res.status(400).json({ ok: false, error: "No se pudo procesar la imagen: " + err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Clasificador de mascotas listo en http://localhost:${PORT}`);
  console.log(`Modelo: ${model.modelo} - ${model.coef.length} descriptores HOG`);
});