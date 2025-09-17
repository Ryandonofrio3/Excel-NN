# Neural Network in Excel

A working neural network that runs entirely inside Excel formulas. Draw in a spreadsheet and watch Excel predict your doodle in real time.

## What This Is

- **784 → 32 → 5** neural network classifier
- Trained on Google's QuickDraw dataset (cats, houses, suns, ladders, trees, doors)
- Runs completely in Excel with no plugins or macros
- 25,000 weights implemented as spreadsheet formulas
- Real-time prediction as you draw

## Files

- `NeuralNet_Excel_Demo2.xlsx` - The main Excel demo (draw and predict)
- `nn_excel_demo.py` - Python training script and Excel export
- `tui/` - Terminal UI version using OpenTUI
- `data/` - QuickDraw dataset (.npy files)

## Quick Start

### Excel Demo
1. Open `NeuralNet_Excel_Demo2.xlsx`
2. Draw in the 28×28 grid (click cells to toggle black/white)
3. Watch predictions update in real-time on the right

### Python Training/Export
```bash
# Install dependencies (requires Python 3.12+)
uv sync

# Run the main script
uv run nn_excel_demo.py
```

### Terminal UI
```bash
cd tui
bun install
bun run dev
```

## How It Works

The neural network uses:
- **784 input neurons** (28×28 pixel grid)
- **32 hidden neurons** with ReLU activation
- **5 output neurons** (one per class)

Each hidden neuron is a single Excel formula:
```excel
=MAX(0, SUMPRODUCT($AC$1:$AC$784, Weights!A1:A784) + Weights!A785)
```

Final predictions use softmax to convert logits to probabilities.

## Limitations

- **Binary input**: Cells are either 0 or 1 (original data has gradients)
- **Fidelity**: 28×28 resolution loses fine detail
- **Accuracy**: ~93% max due to distribution differences

## Requirements

- **Excel**: Any modern version (tested on Excel 365)
- **Python**: 3.12+ with uv package manager
- **Node.js**: Bun runtime for TUI demo

---

*Built with Claude Code in about an hour. Sometimes it's fun to ask AI to do something absurd.*
