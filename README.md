# Text-to-Motion

A C# .NET 8.0 pipeline for processing, training, and visualizing human motion data from the HumanML3D dataset.

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [Preprocessing](#3-preprocessing)
4. [Training](#4-training)
5. [Inference](#5-inference)
6. [Visualization](#6-visualization)

---

## 1. Prerequisites

- [.NET 8.0 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- HumanML3D dataset placed in the `dataset/` folder with the following structure:
  ```
  dataset/
  ├── new_joint_vecs/   (.npy motion files)
  ├── texts/            (caption .txt files)
  ├── train.txt
  ├── val.txt
  ├── test.txt
  ├── Mean.npy
  └── Std.npy
  ```

## 2. Project Structure

| Project | Type | Description |
|---------|------|-------------|
| `text-to-motion` | Console (Exe) | Main entry point, routes to preprocessing/training/inference |
| `Data` | Class Library | Skeleton definition (22 joints), feature slicing (263-dim vectors) |
| `Preprocess` | Class Library | Converts `.npy` + `.txt` into `.bin` format |
| `Train` | Class Library | Training module |
| `Inference` | Class Library | Inference module |
| `Visualize` | ASP.NET Web | Localhost 3D skeleton viewer powered by Three.js |

## 3. Preprocessing

Converts raw HumanML3D `.npy` motion files and `.txt` captions into compact `.bin` files. Also converts normalization statistics (`Mean.npy`, `Std.npy`).

Configure paths in `Configs/preprocessing.json`:
```json
{
  "DatasetPath": "D:/Code/CSharp/text-to-motion/dataset",
  "OutputPath": "D:/Code/CSharp/text-to-motion/processed"
}
```

Run:
```bash
dotnet run --project text-to-motion.csproj -- preprocessing
```

Output is written to:
```
processed/
├── train/   (*.bin sample files)
├── val/
├── test/
├── mean.bin
└── std.bin
```

## 4. Training

```bash
dotnet run --project text-to-motion.csproj -- training
```

Configure in `Configs/training.json`.

## 5. Inference

```bash
dotnet run --project text-to-motion.csproj -- inference
```

Configure in `Configs/inference.json`.

## 6. Visualization

A standalone ASP.NET Minimal API application that serves a browser-based 3D skeleton viewer. Reads the `.bin` files produced by [Preprocessing](#3-preprocessing) and renders animated skeletons using Three.js.

Run:
```bash
cd Visualize
dotnet run --urls "http://localhost:5050"
```

Then open [http://localhost:5050](http://localhost:5050) in your browser.

Features:
- Scrollable animation list with search/filter
- Caption display for each animation
- 3D viewport with orbit controls (rotate, zoom, pan)
- Color-coded skeleton by body group (spine, legs, arms)
- Play/Pause with interactive timeline slider
- 20 FPS playback matching HumanML3D framerate
