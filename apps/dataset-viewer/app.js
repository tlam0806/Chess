const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");
const depthEl = document.getElementById("depth");
const evalWhiteEl = document.getElementById("evalWhite");
const evalTextEl = document.getElementById("evalText");
const undoTurnBtn = document.getElementById("undoTurn");
const undoBtn = document.getElementById("undo");
const resetBtn = document.getElementById("reset");

const pieceGlyph = {
  P: "♟", N: "♞", B: "♝", R: "♜", Q: "♛", K: "♚",
  p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚",
  ".": ""
};

let state = null;
let selected = null;
let lastMove = "";
let busy = false;

function setBusy(nextBusy) {
  busy = nextBusy;
  undoTurnBtn.disabled = nextBusy;
  undoBtn.disabled = nextBusy;
  resetBtn.disabled = nextBusy;
  depthEl.disabled = nextBusy;
}

function squareName(file, rank) {
  return "abcdefgh"[file] + String(rank + 1);
}

function pieceAt(square) {
  const file = square.charCodeAt(0) - "a".charCodeAt(0);
  const rank = Number(square[1]) - 1;
  const row = 7 - rank;
  return state.board[row][file];
}

function legalFrom(square) {
  return state.legal.filter(move => move.startsWith(square));
}

function legalTargets(square) {
  return new Set(legalFrom(square).map(move => move.slice(2, 4)));
}

function addLog(text) {
  const line = document.createElement("div");
  line.textContent = text;
  logEl.prepend(line);
}

function updateEvalBar() {
  const evalCp = state.eval ?? 0;
  const clamped = Math.max(-1000, Math.min(1000, evalCp));
  const whitePercent = 50 + clamped / 20;
  evalWhiteEl.style.height = `${whitePercent}%`;

  if (evalCp > 0) {
    evalTextEl.textContent = `W +${(evalCp / 100).toFixed(2)}`;
  } else if (evalCp < 0) {
    evalTextEl.textContent = `B +${(-evalCp / 100).toFixed(2)}`;
  } else {
    evalTextEl.textContent = "0.00";
  }
}

async function api(path, payload = null) {
  const options = payload
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }
    : {};
  const response = await fetch(path, options);
  return response.json();
}

async function refresh() {
  state = await api("/api/state");
  render();
  maybeBotMove();
}

function render() {
  boardEl.innerHTML = "";
  const targets = selected ? legalTargets(selected) : new Set();
  const lastFrom = lastMove.slice(0, 2);
  const lastTo = lastMove.slice(2, 4);

  for (let rank = 7; rank >= 0; --rank) {
    for (let file = 0; file < 8; ++file) {
      const square = squareName(file, rank);
      const row = 7 - rank;
      const piece = state.board[row][file];

      const cell = document.createElement("button");
      cell.className = `square ${(file + rank) % 2 === 0 ? "dark" : "light"}`;
      cell.dataset.square = square;
      const pieceEl = document.createElement("span");
      pieceEl.className = "piece";
      if (piece >= "A" && piece <= "Z") {
        pieceEl.classList.add("white-piece");
      } else if (piece >= "a" && piece <= "z") {
        pieceEl.classList.add("black-piece");
      }
      pieceEl.textContent = pieceGlyph[piece];
      cell.appendChild(pieceEl);

      if (square === selected) {
        cell.classList.add("selected");
      }
      if (targets.has(square)) {
        cell.classList.add("target");
        if (piece !== ".") {
          cell.classList.add("capture");
        }
      }
      if (square === lastFrom || square === lastTo) {
        cell.classList.add("last");
      }
      if (file === 0) {
        const coord = document.createElement("span");
        coord.className = "coord";
        coord.textContent = String(rank + 1);
        cell.appendChild(coord);
      }

      cell.addEventListener("click", () => onSquareClick(square));
      boardEl.appendChild(cell);
    }
  }

  const statusText = {
    playing: "Playing",
    check: "Check",
    checkmate: "Checkmate",
    stalemate: "Stalemate"
  }[state.status] ?? state.status;

  statusEl.textContent = `${statusText}. Side: ${state.side === "w" ? "White" : "Black"}`;
  updateEvalBar();
}

async function onSquareClick(square) {
  if (busy || !state || state.gameOver || state.side !== "w") {
    return;
  }

  const piece = pieceAt(square);

  if (!selected) {
    if (piece >= "A" && piece <= "Z" && legalFrom(square).length > 0) {
      selected = square;
      render();
    }
    return;
  }

  if (selected === square) {
    selected = null;
    render();
    return;
  }

  const candidates = legalFrom(selected).filter(move => move.slice(2, 4) === square);
  if (candidates.length === 0) {
    if (piece >= "A" && piece <= "Z" && legalFrom(square).length > 0) {
      selected = square;
      render();
    }
    return;
  }

  let move = candidates[0];
  if (candidates.length > 1) {
    move = candidates.find(candidate => candidate.endsWith("q")) ?? move;
  }

  selected = null;
  await makePlayerMove(move);
}

async function makePlayerMove(move) {
  setBusy(true);
  const next = await api("/api/move", { move });
  setBusy(false);

  if (!next.ok) {
    addLog(next.message || "Illegal move");
    return;
  }

  state = next;
  lastMove = move;
  addLog(`White: ${move}`);
  render();
  maybeBotMove();
}

async function maybeBotMove() {
  if (!state || busy || state.gameOver || state.side !== "b") {
    return;
  }

  setBusy(true);
  statusEl.textContent = "Bot thinking...";
  const depth = Number(depthEl.value || 4);
  const next = await api("/api/bot", { depth });
  setBusy(false);

  state = next;
  if (next.lastMove) {
    lastMove = next.lastMove;
    addLog(`Black: ${next.lastMove} (${next.message})`);
  }
  render();
}

resetBtn.addEventListener("click", async () => {
  if (busy) return;
  selected = null;
  lastMove = "";
  state = await api("/api/reset", {});
  logEl.innerHTML = "";
  render();
});

undoBtn.addEventListener("click", async () => {
  if (busy) return;
  selected = null;
  state = await api("/api/undo", {});
  lastMove = "";
  addLog(state.message);
  render();
});

undoTurnBtn.addEventListener("click", async () => {
  if (busy) return;
  selected = null;
  state = await api("/api/undo_turn", {});
  lastMove = "";
  addLog(state.message);
  render();
});

refresh();
