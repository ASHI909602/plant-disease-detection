# Plant Disease Detection

Deep learning model that classifies plant leaf images into **38 disease / healthy classes** using transfer learning (EfficientNetB0, PyTorch), with a Gradio web demo.

**Live demo:** <Hugging Face Space link yahan daalo>

## Dataset
[PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) (`color` folder): 54,305 images, 38 classes, 14 crop species.

## Approach
- Pretrained EfficientNetB0 (ImageNet), final layer replaced for 38 classes
- Augmentation: horizontal flip, rotation (20°), color jitter, resize to 224x224
- 80/10/10 train/validation/test split
- AdamW optimizer, cosine learning-rate schedule, 8 epochs on a Kaggle GPU
- Best checkpoint chosen by validation accuracy

## Results
| Metric | Value |
|---|---|
| Test accuracy | **99.85%** |

![Training curves](images/training_curves.png)
![Confusion matrix](images/confusion_matrix.png)

## Limitations
- PlantVillage images are taken in lab conditions on plain backgrounds, and the dataset contains multiple photos of the same leaf. With a random split, near-duplicate images can appear in both train and test sets, so the test accuracy is likely higher than real-world performance.
- The model only knows these 38 classes and may give confident but wrong answers for other crops or diseases.
- Not a replacement for expert diagnosis.

## Future work
- Fine-tune on field images (e.g. PlantDoc) for better real-world robustness
- Split by leaf/source to avoid near-duplicate leakage
- Add an "unknown" class and confidence thresholding

## Run locally
```bash
pip install -r requirements.txt
python app.py
```
To retrain:
```bash
python train.py --data_dir path/to/plantvillage/color --out_dir .
```

## Project structure
```
train.py            # training, evaluation, plots
app.py              # Gradio demo
requirements.txt
model.pth           # trained weights
classes.json        # class names
metrics.json        # test accuracy
images/             # training curves, confusion matrix
```
