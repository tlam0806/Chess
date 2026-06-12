const boardEl = document.getElementById("datasetBoard");
const statusEl = document.getElementById("datasetStatus");
const evalWhiteEl = document.getElementById("datasetEvalWhite");
const evalTextEl = document.getElementById("datasetEvalText");
const firstBtn = document.getElementById("datasetFirst");
const prevBtn = document.getElementById("datasetPrev");
const playBtn = document.getElementById("datasetPlay");
const nextBtn = document.getElementById("datasetNext");
const lastBtn = document.getElementById("datasetLast");
const slider = document.getElementById("datasetSlider");
const infoEl = document.getElementById("datasetInfo");
const listEl = document.getElementById("datasetList");

const pieceGlyph = {
  P: "♟", N: "♞", B: "♝", R: "♜", Q: "♛", K: "♚",
  p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚",
  ".": ""
};

let samples = [];
let sampleIndex = 0;
let timer = null;
let datasetName = "Dataset";

function squareName(file, rank) {
  return "abcdefgh"[file] + String(rank + 1);
}

function isWhite(piece) {
  return piece >= "A" && piece <= "Z";
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function cpText(cp) {
  if (Math.abs(cp) >= 100000) {
    return cp > 0 ? "M+" : "M-";
  }
  const pawns = cp / 100;
  return `${pawns >= 0 ? "+" : ""}${pawns.toFixed(2)}`;
}

async function loadDataset() {
  const params = new URLSearchParams(window.location.search);
  const view = params.get("view");
  const file = params.get("file");
  const limit = params.get("limit") || "160";
  const endpoint = view
    ? `/api/dataset_view?file=${encodeURIComponent(view)}`
    : `/api/dataset_game?limit=${encodeURIComponent(limit)}${file ? `&file=${encodeURIComponent(file)}` : ""}`;
  const response = await fetch(endpoint);
  const payload = await response.json();
  if (!payload.ok) {
    statusEl.textContent = payload.message || "Failed to load dataset.";
    return;
  }
  datasetName = payload.dataset || "Dataset";
  samples = payload.samples;
  slider.max = String(Math.max(0, samples.length - 1));
  slider.value = "0";
  sampleIndex = 0;
  render();
}

function renderBoard(sample) {
  boardEl.innerHTML = "";
  for (let rank = 7; rank >= 0; --rank) {
    for (let file = 0; file < 8; ++file) {
      const square = squareName(file, rank);
      const piece = sample.board[square] || ".";
      const cell = document.createElement("div");
      cell.className = `square ${(file + rank) % 2 === 0 ? "dark" : "light"}`;

      const pieceEl = document.createElement("span");
      pieceEl.className = "piece";
      if (isWhite(piece)) {
        pieceEl.classList.add("white-piece");
      } else if (piece !== ".") {
        pieceEl.classList.add("black-piece");
      }
      pieceEl.textContent = pieceGlyph[piece] || "";
      cell.appendChild(pieceEl);

      if (file === 0) {
        const coord = document.createElement("span");
        coord.className = "coord";
        coord.textContent = String(rank + 1);
        cell.appendChild(coord);
      }

      boardEl.appendChild(cell);
    }
  }
}

function renderEval(sample) {
  const whiteCp = sample.white_target;
  const height = clamp(50 + whiteCp / 20, 5, 95);
  evalWhiteEl.style.height = `${height}%`;
  evalTextEl.textContent = cpText(whiteCp);
}

function renderList() {
  listEl.innerHTML = "";
  samples.forEach((sample, index) => {
    const row = document.createElement("button");
    row.className = "move-row";
    if (index === sampleIndex) {
      row.classList.add("active");
    }
    const move = sample.move ? ` ${sample.move}` : "";
    row.textContent = `${index + 1}.${move} ${sample.side_to_move} target ${sample.target} white ${sample.white_target}`;
    row.addEventListener("click", () => {
      stopPlayback();
      setSample(index);
    });
    listEl.appendChild(row);
  });
}

function render() {
  if (!samples.length) return;
  const sample = samples[sampleIndex];
  slider.value = String(sampleIndex);
  statusEl.textContent = `${datasetName} Sample ${sampleIndex + 1} / ${samples.length}.`;
  const moveText = sample.move ? `, legal move ${sample.move} from parent ply ${sample.parent_ply}` : "";
  infoEl.textContent =
    `Sample ${sampleIndex + 1}: side ${sample.side_to_move}${moveText}, target ${sample.target} cp from side-to-move, ${sample.white_target} cp from White POV.`;
  renderEval(sample);
  renderBoard(sample);
  renderList();
}

function setSample(index) {
  sampleIndex = clamp(index, 0, samples.length - 1);
  render();
}

function startPlayback() {
  if (timer) return;
  playBtn.textContent = "Pause";
  timer = setInterval(() => {
    if (sampleIndex >= samples.length - 1) {
      stopPlayback();
      return;
    }
    setSample(sampleIndex + 1);
  }, 450);
}

function stopPlayback() {
  if (!timer) return;
  clearInterval(timer);
  timer = null;
  playBtn.textContent = "Play";
}

firstBtn.addEventListener("click", () => {
  stopPlayback();
  setSample(0);
});

prevBtn.addEventListener("click", () => {
  stopPlayback();
  setSample(sampleIndex - 1);
});

nextBtn.addEventListener("click", () => {
  stopPlayback();
  setSample(sampleIndex + 1);
});

lastBtn.addEventListener("click", () => {
  stopPlayback();
  setSample(samples.length - 1);
});

playBtn.addEventListener("click", () => {
  if (timer) {
    stopPlayback();
  } else {
    startPlayback();
  }
});

slider.addEventListener("input", () => {
  stopPlayback();
  setSample(Number(slider.value));
});

loadDataset();
