# Comprehensive Pore Detection System

When assessing microscopy images using Deep Learning (DL), three specific methods were used, these being U-Net, PoreD², and YOLO.
These methods were applied to each of the imaging modalities (Cryo-SEM, AFM, STED and Confocal)

### Deep Learning Models 

**U-Net** essentially categorisizes each pixel in the image and is biomedical imaging.

**PoreD²** basically finds pores by looking at the image at different zoomed in levels and detects what patterns represent pores

**YOLO (You Only Look Once)** attempts to spot each pore as an individual object by drawing a square around it 

### The different imaging techniques

AFM (Atomic Force Microscopy): 1%, 1.5%, and 2% concentrations  
Confocal Microscopy: 0.375% and 1% concentrations  
Cryo-SEM (Scanning Electron Microscopy): magnifications from x1000 up to x60000  
STED (Stimulated Emission Depletion): 0.375% and 1% concentrations  

## How to Run this analysis

First, install all the required packages:
```bash
pip install -r requirements.txt
```

### Training All Models
To train all three models at once:
```bash
python train_all_models.py --data_dir "path/to/Dataset" --epochs 50
```

### Training Each Model
You can train each model on its own.

U-Net:
```bash
python train_unet.py --data_dir "path/to/Dataset" --image_type AFM --epochs 100
```

PoreD²:
```bash
python train_pored2.py --data_dir "path/to/Dataset" --image_type CONFOCAL --epochs 150
```

YOLO:
```bash
python train_yolo.py --data_dir "path/to/Dataset" --image_type CRYO-SEM --epochs 100
```

### Running the System
After the models have went through each of the training images we can use each of our deep learning models on the images we want to test on
```bash
python inference.py --model_type unet --image_type AFM --input_image "path/to/image.tif"
python inference.py --model_type pored2 --image_type CONFOCAL --input_image "path/to/image.tif"
python inference.py --model_type yolo --image_type CRYO-SEM --input_image "path/to/image.tif"
```

### Comparison of our Deep Learning Models Results
After this we perform a simple comparison of the results visually
```bash
python test_and_analyze.py --test_dir "path/to/Dataset" --models unet pored2 yolo
```


## Output

The outputted results show a binary 8-bit image 0-255 with also model time values calculated for each model and also assesses average pore size


## Requirements

When doing this analysis one of the biggets issues was ensuring enough storage was available (recommend using a USB stick or getting extra storage through a google drive plan) and Python 3.11.  

### Main Libraries
PyTorch 2.0+  
OpenCV 4.8+  
NumPy, Matplotlib, Pillow  
Ultralytics (for YOLO)  
Albumentations (for image augmentation)  
scikit-image, seaborn, pandas  
The rest can be found on requirements.txt file 

## Method

The labels for each of the images were made by Ilastik which is described in the Methods Section of the thesis.
These labels essentially mark where pores are likely to be found. However due to limited data size overfitting was a big issue so in order to minimise overfitting data augmentation was used during training (flips, rotations, brightness changes) Models are tested with cross-validation to check how well they generalize.

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI 2015*.  
2. Karaca, I., et al. (2023). PoreD²: Multi-task learning architecture for pore detection and analysis. *GitHub repository*: https://github.com/ilaydakaraca/PoreD2  
3. Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics  
4. Woo, S. et al. (2018). CBAM: Convolutional block attention module. *ECCV 2018*.  
5. Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data*.
