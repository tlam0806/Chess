const boardEl = document.getElementById("replayBoard");
const statusEl = document.getElementById("replayStatus");
const replaySelect = document.getElementById("replaySelect");
const firstBtn = document.getElementById("firstMove");
const prevBtn = document.getElementById("prevMove");
const playBtn = document.getElementById("playPause");
const nextBtn = document.getElementById("nextMove");
const lastBtn = document.getElementById("lastMove");
const slider = document.getElementById("plySlider");
const moveInfo = document.getElementById("moveInfo");
const moveList = document.getElementById("moveList");

const pieceGlyph = {
  P: "♟", N: "♞", B: "♝", R: "♜", Q: "♛", K: "♚",
  p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚",
  ".": ""
};

const startRows = [
  "rnbqkbnr",
  "pppppppp",
  "........",
  "........",
  "........",
  "........",
  "PPPPPPPP",
  "RNBQKBNR"
];

let replay = null;
let board = null;
let plyIndex = 0;
let lastMove = "";
let timer = null;

function cloneStartBoard() {
  return startRows.map(row => row.split(""));
}

function squareName(file, rank) {
  return "abcdefgh"[file] + String(rank + 1);
}

function squareToCoord(square) {
  const file = square.charCodeAt(0) - "a".charCodeAt(0);
  const rank = Number(square[1]) - 1;
  return { row: 7 - rank, file, rank };
}

function getPiece(nextBoard, square) {
  const { row, file } = squareToCoord(square);
  return nextBoard[row][file];
}

function setPiece(nextBoard, square, piece) {
  const { row, file } = squareToCoord(square);
  nextBoard[row][file] = piece;
}

function isWhite(piece) {
  return piece >= "A" && piece <= "Z";
}

function promotionPiece(piece, promotion) {
  if (!promotion) return piece;
  return isWhite(piece) ? promotion.toUpperCase() : promotion.toLowerCase();
}

function applyMove(nextBoard, move, epSquare) {
  const from = move.slice(0, 2);
  const to = move.slice(2, 4);
  const promotion = move[4] || "";
  const piece = getPiece(nextBoard, from);
  const target = getPiece(nextBoard, to);
  let nextEpSquare = "";

  if (piece === ".") {
    throw new Error(`empty source square for ${move}`);
  }

  setPiece(nextBoard, from, ".");

  if ((piece === "K" || piece === "k") && Math.abs(squareToCoord(from).file - squareToCoord(to).file) === 2) {
    if (to === "g1") {
      setPiece(nextBoard, "h1", ".");
      setPiece(nextBoard, "f1", "R");
    } else if (to === "c1") {
      setPiece(nextBoard, "a1", ".");
      setPiece(nextBoard, "d1", "R");
    } else if (to === "g8") {
      setPiece(nextBoard, "h8", ".");
      setPiece(nextBoard, "f8", "r");
    } else if (to === "c8") {
      setPiece(nextBoard, "a8", ".");
      setPiece(nextBoard, "d8", "r");
    }
  }

  if ((piece === "P" || piece === "p") && target === "." && to === epSquare && from[0] !== to[0]) {
    const capturedRank = piece === "P" ? Number(to[1]) - 1 : Number(to[1]) + 1;
    setPiece(nextBoard, `${to[0]}${capturedRank}`, ".");
  }

  const fromRank = Number(from[1]);
  const toRank = Number(to[1]);
  if (piece === "P" && fromRank === 2 && toRank === 4) {
    nextEpSquare = `${from[0]}3`;
  } else if (piece === "p" && fromRank === 7 && toRank === 5) {
    nextEpSquare = `${from[0]}6`;
  }

  setPiece(nextBoard, to, promotionPiece(piece, promotion));
  return nextEpSquare;
}

function buildBoardAt(index) {
  const nextBoard = cloneStartBoard();
  let epSquare = "";
  for (let i = 0; i < index; ++i) {
    epSquare = applyMove(nextBoard, replay.moves[i].move, epSquare);
  }
  return nextBoard;
}

async function api(path) {
  const response = await fetch(path);
  return response.json();
}

async function loadReplayList() {
  const payload = await api("/api/replays");
  if (!payload.ok || payload.replays.length === 0) {
    statusEl.textContent = "No replay files found in data/matches.";
    return;
  }

  replaySelect.innerHTML = "";
  for (const file of payload.replays) {
    const option = document.createElement("option");
    option.value = file;
    option.textContent = file.replace(".txt", "");
    replaySelect.appendChild(option);
  }

  replaySelect.addEventListener("change", () => loadReplay(replaySelect.value));
  await loadReplay(payload.replays[0]);
}

async function loadReplay(file) {
  stopPlayback();
  replay = await api(`/api/replay?file=${encodeURIComponent(file)}`);
  if (!replay.ok) {
    statusEl.textContent = replay.message || "Failed to load replay.";
    return;
  }
  plyIndex = 0;
  slider.max = String(replay.moves.length);
  slider.value = "0";
  render();
}

function renderBoard() {
  board = buildBoardAt(plyIndex);
  boardEl.innerHTML = "";
  const lastFrom = lastMove.slice(0, 2);
  const lastTo = lastMove.slice(2, 4);

  for (let rank = 7; rank >= 0; --rank) {
    for (let file = 0; file < 8; ++file) {
      const square = squareName(file, rank);
      const row = 7 - rank;
      const piece = board[row][file];

      const cell = document.createElement("div");
      cell.className = `square ${(file + rank) % 2 === 0 ? "dark" : "light"}`;
      if (square === lastFrom || square === lastTo) {
        cell.classList.add("last");
      }

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

function renderMoveList() {
  moveList.innerHTML = "";
  replay.moves.forEach((move, index) => {
    const row = document.createElement("button");
    row.className = "move-row";
    if (index === plyIndex - 1) {
      row.classList.add("active");
    }
    row.textContent = `${move.ply}. ${move.side} ${move.engine} ${move.move} score ${move.score}`;
    row.addEventListener("click", () => {
      stopPlayback();
      setPly(index + 1);
    });
    moveList.appendChild(row);
  });
}

function renderInfo() {
  const header = replay.header;
  const current = plyIndex > 0 ? replay.moves[plyIndex - 1] : null;
  lastMove = current ? current.move : "";
  statusEl.textContent =
    `Depth ${header.depth}. White: ${header.white}. Black: ${header.black}. Result: ${header.result} (${header.reason}).`;

  if (!current) {
    moveInfo.textContent = "Start position";
  } else {
    moveInfo.textContent =
      `Ply ${current.ply}: ${current.side} ${current.engine} played ${current.move}, score ${current.score}, nodes ${current.nodes}`;
  }
}

function render() {
  if (!replay) return;
  slider.value = String(plyIndex);
  renderInfo();
  renderBoard();
  renderMoveList();
}

function setPly(nextIndex) {
  plyIndex = Math.max(0, Math.min(replay.moves.length, nextIndex));
  render();
}

function startPlayback() {
  if (timer || !replay) return;
  playBtn.textContent = "Pause";
  timer = setInterval(() => {
    if (plyIndex >= replay.moves.length) {
      stopPlayback();
      return;
    }
    setPly(plyIndex + 1);
  }, 650);
}

function stopPlayback() {
  if (!timer) return;
  clearInterval(timer);
  timer = null;
  playBtn.textContent = "Play";
}

firstBtn.addEventListener("click", () => {
  stopPlayback();
  setPly(0);
});

prevBtn.addEventListener("click", () => {
  stopPlayback();
  setPly(plyIndex - 1);
});

nextBtn.addEventListener("click", () => {
  stopPlayback();
  setPly(plyIndex + 1);
});

lastBtn.addEventListener("click", () => {
  stopPlayback();
  setPly(replay.moves.length);
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
  setPly(Number(slider.value));
});

loadReplayList();
