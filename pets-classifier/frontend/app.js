const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const dzEmpty = document.getElementById("dz-empty");
const dzPreview = document.getElementById("dz-preview");
const previewImg = document.getElementById("preview-img");
const btnCambiar = document.getElementById("btn-cambiar");
const resultado = document.getElementById("resultado");
const historia = document.getElementById("historia");
const listaHistoria = document.getElementById("lista-historia");
const confettiCapa = document.getElementById("confetti");

const emojiEl = document.getElementById("emoji");
const tituloEl = document.getElementById("r-titulo");
const descEl = document.getElementById("r-descripcion");
const rellenoGato = document.getElementById("relleno-gato");
const rellenoPerro = document.getElementById("relleno-perro");
const valGato = document.getElementById("val-gato");
const valPerro = document.getElementById("val-perro");
const rMs = document.getElementById("r-ms");
const rRes = document.getElementById("r-resolucion");
const rZ = document.getElementById("r-z");

let imagenActual = null;
let ocupado = false;

const EMOJI = { Perro: "🐶", Gato: "🐱" };
const COLOR = { Perro: "#fb923c", Gato: "#38bdf8" };
const FRASES = {
  alta: "El modelo está muy seguro de esta decisión.",
  media: "El modelo cree esto, pero no está demasiado seguro.",
];

function mostrarVistaPrevia(dataUrl) {
  previewImg.src = dataUrl;
  dzEmpty.classList.add("hidden");
  dzPreview.classList.remove("hidden");
}

function resetVista() {
  previewImg.removeAttribute("src");
  dzPreview.classList.add("hidden");
  dzEmpty.classList.remove("hidden");
}

function empezarClasificacion(dataUrl) {
  if (ocupado) return;
  ocupado = true;
  mostrarVistaPrevia(dataUrl);
  resultado.classList.add("hidden");

  moverBarras(0, 0, 0, 0);

  fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl }),
  })
    .then((r) => r.json())
    .then((res) => {
      if (!res.ok) throw new Error(res.error || "Error desconocido");
      pintarResultado(res);
      guardarEnHistoria(dataUrl, res.clase, res.probClase);
      imagenActual = { dataUrl, clase: res.clase, prob: res.probClase };
    })
    .catch((err) => {
      resultado.classList.remove("hidden");
      vaciarBarras();
      tituloEl.textContent = "Ups, error";
      emojiEl.textContent = "🤖";
      descEl.textContent = err.message || "No se pudo procesar la imagen.";
    })
    .finally(() => {
      ocupado = false;
    });
}

function pintarResultado(res) {
  const esPerro = res.clase === "Perro";
  const probClase = res.probClase;
  const probPerro = esPerro ? probClase : 1 - probClase;
  const probGato = 1 - probPerro;

  emojiEl.textContent = EMOJI[res.clase];
  tituloEl.textContent = res.clase;
  tituloEl.style.color = COLOR[res.clase];
  descEl.textContent = probClase >= 0.65 ? FRASES.alta : FRASES.media;

  moverBarras(probGato, probPerro, probGato * 100, probPerro * 100);

  rMs.textContent = `⚡ ${res.ms} ms`;
  rRes.textContent = res.imagen ? `📐 ${res.imagen.ancho}×${res.imagen.alto} px` : "";
  rZ.textContent = res.z != null ? `z = ${res.z.toFixed(3)}` : "";

  resultado.classList.remove("hidden");
  resultado.scrollIntoView({ behavior: "smooth", block: "nearest" });

  if (probClase >= 0.8) {
    lanzarConfeti();
  }
}

function moverBarras(gato, perro, gatoPct, perroPct) {
  rellenoGato.style.width = gato * 100 + "%";
  rellenoPerro.style.width = perro * 100 + "%";
  animarContador(valGato, gatoPct);
  animarContador(valPerro, perroPct);
}

function vaciarBarras() {
  rellenoGato.style.width = "0%";
  rellenoPerro.style.width = "0%";
  valGato.textContent = "0%";
  valPerro.textContent = "0%";
}

function animarContador(el, objetivo) {
  const inicio = performance.now();
  const desde = 0;
  const duracion = 600;
  function tick(ahora) {
    const t = Math.min((ahora - inicio) / duracion, 1);
    const suavizado = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(desde + (objetivo - desde) * suavizado) + "%";
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function guardarEnHistoria(dataUrl, clase, prob) {
  historia.classList.remove("hidden");
  const item = document.createElement("div");
  item.className = "hitem";
  const img = document.createElement("img");
  img.src = dataUrl;
  const label = document.createElement("span");
  label.className = "h-label";
  label.textContent = `${EMOJI[clase]} ${Math.round(prob * 100)}%`;
  label.style.color = COLOR[clase];
  img.addEventListener("click", () => empezarClasificacion(dataUrl));
  item.append(img, label);
  listaHistoria.prepend(item);
  while (listaHistoria.children.length > 12) {
    listaHistoria.lastElementChild.remove();
  }
}

function lanzarConfeti() {
  const colores = ["#38bdf8", "#fb923c", "#a78bfa", "#4ade80", "#facc15", "#f472b6"];
  for (let i = 0; i < 70; i++) {
    const c = document.createElement("div");
    c.className = "confeti";
    c.style.left = Math.random() * 100 + "vw";
    c.style.width = 6 + Math.random() * 8 + "px";
    c.style.height = 10 + Math.random() * 10 + "px";
    c.style.background = colores[Math.floor(Math.random() * colores.length)];
    c.style.animationDuration = 2.4 + Math.random() * 2.2 + "s";
    c.style.animationDelay = Math.random() * 0.6 + "s";
    confettiCapa.appendChild(c);
    setTimeout(() => c.remove(), 6500);
  }
}

dropzone.addEventListener("click", (e) => {
  if (e.target.id === "btn-cambiar") return;
  fileInput.click();
});

btnCambiar.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  const archivo = fileInput.files[0];
  if (archivo) leerArchivo(archivo);
  fileInput.value = "";
});

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("arrastrando");
  })
);

["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("arrastrando");
  })
);

dropzone.addEventListener("drop", (e) => {
  const archivo = e.dataTransfer.files[0];
  if (archivo) leerArchivo(archivo);
});

document.addEventListener("paste", (e) => {
  const items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  for (const item of items) {
    if (item.type && item.type.startsWith("image/")) {
      const archivo = item.getAsFile();
      if (archivo) leerArchivo(archivo);
      return;
    }
  }
});

function leerArchivo(archivo) {
  if (archivo.type && !archivo.type.startsWith("image/")) return;
  const lector = new FileReader();
  lector.onload = () => empezarClasificacion(lector.result);
  lector.readAsDataURL(archivo);
}

vaciarBarras();