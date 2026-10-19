# Plant Disease Classifier

![CI](https://github.com/taha2554/plant-disease-classifier/actions/workflows/ci.yml/badge.svg)

Classifies plant diseases from photos of leaves using **transfer learning in PyTorch**. An ImageNet-pretrained CNN (MobileNetV3-Small by default, ResNet-18 or EfficientNet-B0 optional) gets a new classification head and is fine-tuned on the [PlantVillage](https://data.mendeley.com/datasets/tywbtsjrjv/1) dataset (≈54k images, 38 classes across 14 crops).

## Training approach

1. **Stratified split** – train / validation / test (80/10/10) with the same class proportions in each part, fixed seed, no files copied.
2. **Augmentation** – random crops, flips and rotations; colour jitter is kept small because colour (yellowing, brown spots) is part of the symptom.
3. **Two-phase fine-tuning**
   - phase 1: backbone frozen, only the new head is trained (higher learning rate)
   - phase 2: whole network unfrozen, learning rate ÷10 with cosine decay
4. **Class imbalance** – PlantVillage classes range from ~150 to ~5,000 images, so the loss uses inverse-frequency class weights, plus light label smoothing.
5. **Evaluation** – the checkpoint with the best validation accuracy is evaluated once on the untouched test split; accuracy and per-class F1 are written to `models/report.json`.

Training picks CUDA, Apple Silicon (MPS) or CPU automatically.

## Usage

```bash
pip install -e ".[dev]"
```

Download PlantVillage (e.g. from Mendeley Data or Kaggle) and arrange it as one folder per class:

```
data/plantvillage/
  Apple___Apple_scab/
  Apple___healthy/
  Tomato___Early_blight/
  ...
```

Train:

```bash
python -m plant_disease.train --data data/plantvillage --epochs 8
python -m plant_disease.train --data data/plantvillage --backbone resnet18 --batch-size 64
```

Predict:

```bash
python -m plant_disease.predict my_leaf.jpg --top 3
```

This prints the three most likely classes with their probabilities, with the folder names turned into readable labels (`Tomato___Early_blight` → `Tomato – Early blight`).

## Tests

```bash
pytest
```

The tests generate a tiny synthetic dataset (green leaves vs. leaves with brown spots) and run the whole pipeline end to end – split, loaders, two-phase training, checkpoint, prediction – on CPU in a few seconds, without downloading weights or data.

## Notes

- PlantVillage photos are taken on a plain background in a lab. A model trained only on them usually performs worse on field photos with soil, hands and other leaves in the frame; collecting some real-world images for the test set is the next step.
- No trained weights are included in the repository (they're large and dataset-specific).

## Project layout

```
src/plant_disease/data.py     transforms, stratified split, loaders, class weights
src/plant_disease/model.py    backbones, freezing helpers, checkpoint save/load
src/plant_disease/train.py    training loop and evaluation
src/plant_disease/predict.py  CLI + LeafClassifier class
tests/                        end-to-end smoke tests
```

## License

MIT (code). PlantVillage is published by Hughes & Salathé under CC BY 4.0.
