NN OpenTUI Demo

Draw a 28×28 digit-like sketch in your terminal and see live predictions from the tiny NN, using @opentui/core.

Prereqs
- Bun >= 1.2 (OpenTUI targets Bun runtime)
- Model JSON exported from the Excel demo

Export model JSON
1) Generate the Excel workbook (or reuse your existing):
   python nn_excel_demo.py --out NeuralNet_Excel_Demo2.xlsx --per-class 2000 --epochs 10
2) Export weights to JSON:
   python export_model_json.py NeuralNet_Excel_Demo2.xlsx --out NeuralNet_Excel_Demo2.json

Install deps
   cd tui
   bun install

Run
   bun run src/index.ts ../NeuralNet_Excel_Demo2.json

Controls
- Mouse: click/drag to paint
- c: clear
- q: quit

