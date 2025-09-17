import { createCliRenderer, BoxRenderable, TextRenderable, getKeyHandler } from "@opentui/core";
import fs from "node:fs";

type ModelJSON = {
  W1: number[][]; // [784][32]
  b1: number[];   // [32]
  W2: number[][]; // [32][5]
  b2: number[];   // [5]
  labels: string[]; // [5]
};

function relu(x: number): number { return x > 0 ? x : 0; }

function softmax(logits: number[]): number[] {
  const m = Math.max(...logits);
  const e = logits.map(v => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / (s || 1));
}

function predict(model: ModelJSON, x784: number[]): { logits: number[]; probs: number[]; pred: number } {
  const hidden = new Array(model.b1.length).fill(0);
  for (let j = 0; j < model.b1.length; j++) {
    let s = model.b1[j];
    for (let i = 0; i < 784; i++) s += x784[i] * model.W1[i][j];
    hidden[j] = relu(s);
  }
  const logits = new Array(model.b2.length).fill(0);
  for (let k = 0; k < model.b2.length; k++) {
    let s = model.b2[k];
    for (let j = 0; j < hidden.length; j++) s += hidden[j] * model.W2[j][k];
    logits[k] = s;
  }
  const probs = softmax(logits);
  let pred = 0; let best = probs[0];
  for (let k = 1; k < probs.length; k++) if (probs[k] > best) { best = probs[k]; pred = k; }
  return { logits, probs, pred };
}

// Cap input values like the Excel version - allow weighted input up to 3
function clampInput(n: number): number { 
  return Math.max(0, Math.min(3, n)); 
}

async function main() {
  const jsonPath = process.argv[2] || "../NeuralNet_Excel_Demo2.json";
  if (!fs.existsSync(jsonPath)) {
    console.error(`Model JSON not found: ${jsonPath}`);
    console.error("Usage: bun run src/index.ts <path-to-model.json>");
    process.exit(1);
  }
  const model: ModelJSON = JSON.parse(fs.readFileSync(jsonPath, "utf8"));

  const renderer = await createCliRenderer({ useMouse: true, useAlternateScreen: true, useConsole: false });

  const root = renderer.root;

  // Layout: Left grid (28x28), Right panel with predictions & help.
  const app = new BoxRenderable(renderer, {
    id: "app",
    width: "100%",
    height: "100%",
    flexDirection: "row",
    padding: 1,
    gap: 2,
    backgroundColor: "#000000",
  });
  root.add(app);

  // Drawing state
  const rows = 28, cols = 28;
  const grid: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));
  
  // Cursor position for keyboard navigation
  let cursorRow = 14, cursorCol = 14; // Start in center
  let currentBrushValue = 1; // Binary mode: always 1

  const gridContainer = new BoxRenderable(renderer, {
    id: "gridContainer",
    border: true,
    borderColor: "#666666",
    padding: 1,
    gap: 0,
    flexDirection: "column",
    title: `Draw: 1-3 keys (intensity), arrows (move), space (paint), c (clear), q (quit) | Brush: ${currentBrushValue}`,
    titleAlignment: "left",
  });
  app.add(gridContainer);

  const cellSize = 2; // width in chars
  
  // Function to get background color based on cell value
  function getCellBg(value: number, isSelected = false): string {
    if (isSelected) return "#FF6B6B"; // Bright red for cursor
    if (value === 0) return "#111111"; // Dark for empty
    if (value === 1) return "#555555"; // Medium gray for value 1
    if (value === 2) return "#AAAAAA"; // Light gray for value 2
    if (value >= 3) return "#FFFFFF"; // White for value 3+
    return "#111111";
  }

  // Build a 28-row container of rows, each with 28 cells
  const cellBoxes: BoxRenderable[][] = [];
  for (let r = 0; r < rows; r++) {
    const rowBox = new BoxRenderable(renderer, { id: `row-${r}`, flexDirection: "row" });
    gridContainer.add(rowBox);
    const rowCells: BoxRenderable[] = [];
    for (let c = 0; c < cols; c++) {
      const cell = new BoxRenderable(renderer, {
        id: `cell-${r}-${c}`,
        width: cellSize,
        height: 1,
        backgroundColor: getCellBg(0, r === cursorRow && c === cursorCol),
        selectable: true,
        onMouseDown(event) {
          if (event.button !== 0) return;
          // Update cursor position to clicked cell
          cursorRow = r;
          cursorCol = c;
          // Paint with current brush value
          grid[r][c] = grid[r][c] === 0 ? currentBrushValue : 0;
          updateGrid();
          requestUpdate();
        },
        onMouseDrag(event) {
          if (event.button !== 0) return;
          // Update cursor and paint during drag
          cursorRow = r;
          cursorCol = c;
          if (grid[r][c] === 0) {
            grid[r][c] = currentBrushValue;
            updateGrid();
            requestUpdate();
          }
        },
      });
      rowBox.add(cell);
      rowCells.push(cell);
    }
    cellBoxes.push(rowCells);
  }

  // Right panel: predictions & bars
  const side = new BoxRenderable(renderer, {
    id: "side",
    border: true,
    borderColor: "#666666",
    padding: 1,
    gap: 1,
    flexGrow: 1,
    flexDirection: "column",
    title: "Prediction",
  });
  app.add(side);

  const labelText = new TextRenderable(renderer, { id: "pred-label", content: "Label: -" });
  side.add(labelText);

  const probsHeader = new TextRenderable(renderer, { id: "probs-header", content: "Probs:" });
  side.add(probsHeader);

  const probLines: TextRenderable[] = [];
  for (let i = 0; i < model.labels.length; i++) {
    const tr = new TextRenderable(renderer, { id: `prob-${i}`, content: "" });
    side.add(tr);
    probLines.push(tr);
  }

  const help = new TextRenderable(renderer, {
    id: "help",
    content: "1-3: brush intensity | Arrows: move cursor\nSpace: paint | Mouse: click/drag | c: clear | q: quit",
  });
  side.add(help);

  // Function to update grid visual representation
  function updateGrid(): void {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const isSelected = r === cursorRow && c === cursorCol;
        cellBoxes[r][c].backgroundColor = getCellBg(grid[r][c], isSelected);
      }
    }
    // Update title with current brush value
    gridContainer.title = `Draw: 1-3 keys (intensity), arrows (move), space (paint), c (clear), q (quit) | Brush: ${currentBrushValue}`;
  }

  function flattenGrid(): number[] {
    const out = new Array(784);
    let k = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) out[k++] = clampInput(grid[r][c]);
    }
    return out;
  }

  function bar(p: number, width = 24): string {
    const n = Math.max(0, Math.min(width, Math.round(p * width)));
    return "█".repeat(n) + "░".repeat(width - n);
  }

  function updatePrediction(): void {
    const x = flattenGrid();
    const { probs, pred } = predict(model, x);
    labelText.content = `Label: ${model.labels[pred]} (${(probs[pred] * 100).toFixed(1)}%)`;

    // Render each prob as a bar
    for (let i = 0; i < model.labels.length; i++) {
      probLines[i].content = `${model.labels[i].padEnd(8)} ${bar(probs[i])} ${(probs[i] * 100).toFixed(1)}%`;
    }
  }

  let updateQueued = false;
  function requestUpdate() {
    if (updateQueued) return;
    updateQueued = true;
    requestAnimationFrame(() => {
      updateQueued = false;
      updatePrediction();
    });
  }

  // Set up keyboard handling
  const keyHandler = getKeyHandler();
  keyHandler.on("keypress", (key) => {
    let needsUpdate = false;
    let needsGridUpdate = false;

    // Handle movement keys
    if (key.name === "up" || key.name === "w") {
      cursorRow = Math.max(0, cursorRow - 1);
      needsGridUpdate = true;
    } else if (key.name === "down" || key.name === "s") {
      cursorRow = Math.min(rows - 1, cursorRow + 1);
      needsGridUpdate = true;
    } else if (key.name === "left" || key.name === "a") {
      cursorCol = Math.max(0, cursorCol - 1);
      needsGridUpdate = true;
    } else if (key.name === "right" || key.name === "d") {
      cursorCol = Math.min(cols - 1, cursorCol + 1);
      needsGridUpdate = true;
    }
    
    // Handle brush intensity keys
    else if (key.name === "1") {
      currentBrushValue = 1;
      needsGridUpdate = true;
    } else if (key.name === "2") {
      currentBrushValue = 2;
      needsGridUpdate = true;
    } else if (key.name === "3") {
      currentBrushValue = 3;
      needsGridUpdate = true;
    }
    
    // Handle painting
    else if (key.name === "space") {
      const currentValue = grid[cursorRow][cursorCol];
      grid[cursorRow][cursorCol] = currentValue === 0 ? currentBrushValue : 0;
      needsGridUpdate = true;
      needsUpdate = true;
    }
    
    // Handle clear
    else if (key.name === "c") {
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          grid[r][c] = 0;
        }
      }
      needsGridUpdate = true;
      needsUpdate = true;
    }
    
    // Handle quit
    else if (key.name === "q") {
      renderer.stop();
      renderer.destroy();
      process.exit(0);
    }

    if (needsGridUpdate) {
      updateGrid();
    }
    if (needsUpdate) {
      requestUpdate();
    }
  });

  // Initial setup
  updateGrid(); // Initialize grid visuals with cursor
  requestUpdate(); // Initial prediction

  renderer.start();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

