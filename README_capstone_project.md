
# Semantic Segmentation for Indian Road Scenes using Domain Adaptation

This repository contains the complete pipeline for training a semantic segmentation model that performs well on Indian road scenes by applying domain adaptation techniques. The project uses the Mapillary Vistas dataset as the source and the Indian Driving Dataset (IDD) as the target.

## Overview

Semantic segmentation models trained on one domain often fail to generalize to another. This is especially true for Indian road scenes due to different environmental and structural characteristics compared to datasets collected in other parts of the world. The goal of this project is to bridge that gap by training a model using domain adaptation methods.

## Dataset Preparation

### Target Dataset: Indian Driving Dataset (IDD)

1. Visit http://idd.insaan.iiit.ac.in/ and create an account.
2. Download the following files:
   - IDD Segmentation Part I
   - IDD Segmentation Part II
3. Extract both files into the same directory.
4. Generate segmentation masks using the provided script:
   ```bash
   python preperation/createLabels.py --datadir <idd_data_path> --id-type level3Id --num-workers <num_threads>
   ```

### Source Dataset: Mapillary Vistas

1. Download the dataset from the official website: https://www.mapillary.com/dataset/vistas?pKey=q0GhQpk20wJm1ba1mfwJmw
2. Prepare the data using:
   ```bash
   ./domain_adaptation/source/prep_all.sh
   ```
3. The script will generate images and labels in:
   ```
   domain_adaptation/source/source_datasets_dir/
   ```

## Model Architecture

The model is based on the U-Net architecture, implemented using TensorFlow and Keras. It consists of an encoder-decoder structure with skip connections that preserve spatial information. The model is trained using the Adam optimizer and a sparse categorical crossentropy loss function suitable for multi-class pixel-level classification.

## Training and Evaluation

Training and evaluation steps are documented in the `Modern_ML.ipynb` notebook.

### Training Example

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, validation_data=val_ds, epochs=10)
```

### Evaluation

To evaluate the model's segmentation performance, use the following command:

```bash
python evaluate/evaluate_mIoU.py --gts <ground_truth_path> --preds <prediction_path> --num-workers <num_threads>
```

- Predictions and ground truth images should:
  - Be in PNG format
  - Have a resolution of 1280x720
  - Use level3Id for label encoding

## Results

The model achieved the following performance metrics:

- Validation Accuracy: approximately 75.9%
- Validation Loss: approximately 0.91

Visual inspection confirmed that the model accurately segmented various objects in diverse road scenes.

## Tools and Technologies

- Python: scripting and logic
- TensorFlow / Keras: model development
- OpenCV: image processing
- Matplotlib: visualization
- JupyterLab: interactive development
- GitHub: version control

## Challenges and Solutions

Format inconsistencies
Unified different image sizes and label formats using custom preprocessing scripts.

Domain differences
Handled visual and environmental differences by applying domain adaptation techniques.

Label variations
Used official level3Id remapping to ensure consistent label interpretation across datasets.

High training time
Reduced image resolution and optimized batch sizes to speed up training.

Output format requirements
Automated resizing and renaming of predictions to match evaluation standards.

## Repository Structure

```
project_root/
├── preperation/
│   └── createLabels.py
├── domain_adaptation/
│   └── source/
│       └── prep_all.sh
├── evaluate/
│   └── evaluate_mIoU.py
├── models/
│   └── unet_model.py
├── datasets/
│   └── (IDD and Mapillary data)
├── Modern_ML.ipynb
└── README.md
```

## Conclusion

This project demonstrates how domain adaptation can significantly improve the generalization of semantic segmentation models. By training a U-Net model on the Mapillary dataset and adapting it to Indian road scenes, we achieved strong results and meaningful segmentation performance in a real-world context.

## References

- Indian Driving Dataset (IDD): http://idd.insaan.iiit.ac.in/
- Mapillary Vistas: https://www.mapillary.com/dataset/vistas
- U-Net Architecture: https://arxiv.org/abs/1505.04597
