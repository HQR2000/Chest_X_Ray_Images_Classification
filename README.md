# Chest X Ray Images Classification

![Chesr X-ray](https://github.com/HQR2000/Chest_X_Ray_Images_Classification/blob/main/Public/Chest_X_Ray.png)

## Abstract

In this project, we applied deep learning method and transfer learning for the classification of COVID-19 chest x-ray images, virus but not COVID-19 chest x-ray images and normal chest images. Due to the variability of different images and low visibility of the chest x-ray images, we also applied data enhancement method called **White Balance** and **CLAHE** to enhance the quality of the dataset. We divided the classification task to two subtasks, the first one is two do binary classification and the second one is to do multiple classification. The notebook includes all the necessary code for this project can be found in `./code/CXR_classification.ipynb`.

## Dataset

The dataset we used is from Kaggle, you can download the dataset from [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

This dataset contains totally 1823 chest x-ray images collected from different sources and research papers.

There are `536` Covid-19 chest x-ray images, '619' Virus but not Covid-19 chest x-ray images and `668` normal chest x-ray images in the dataset.

## Models

The deep learning models we used for the classification are [NASNetMobile](https://keras.io/api/applications/nasnet/#nasnetmobile-function), [ResNet-101](https://keras.io/api/applications/resnet/#resnet101-function) and [InceptionResNetV2](https://keras.io/api/applications/inceptionresnetv2/).

## Methods

### Data Enhancement

During our training process of the deep learning models, we noticed that the models can hardly gained a good accuracy, therefore we look into the images manually and noticed that most of the images from the dataset may be taken from different x-ray machines so the variability of the images are complex. Besides, the visibility is also low due to the lighting condition. In this case, we implemented two data enhancement method to enhance the quality of the dataset.

#### White Balance

White balance is an image preprocessing methodology used to adjust the appropriate color fidelity of an image. By stretching the RGB channels respectively, white balance algorithm is able to adjust the color of the active layers. While stretching is performed for the color range, the pixel colors at the end of the three channels are discarded since only 0.05% of these pixels are meaningful in the images, after this operation, pixel colors that do not often appear at the end of the channel do not negatively affect the upper and lower limits when stretching is performing.

Here shows the core steps for White Balance algorithm:

![White Balance](https://github.com/HQR2000/Chest_X_Ray_Images_Classification/blob/main/Public/Whtie_Balance.png)

#### CLAHE

CLAHE is an effective contrast enhancement method which can effectively increase the contrast of an image, it's recognized as the updated version of the Adaptive Histogram Equation(AHE). Histogram equalization is an easy way to enhance the contrast of an image by extending the intensity range of the image or the intensity value of the image most frequently. The tensile strength value affects the natural brightness of the input image and add some unnecessary noise into the image. In AHE, the input image is divided into several small images, which are also called tiles. After histograms of each tile which corresponds to different parts of the image are calculated, they are used for the derivative of the intensity mapping function. AHE add extra noise caused by over amplification into the image. CLAHE works in the same way as AHE while it clips the histogram with specific values to limit magnification before calculating the cumulative distribution function.

The formula for CLAHE is as below:

![CLAHE](https://github.com/HQR2000/Chest_X_Ray_Images_Classification/blob/main/Public/CLAHE.png)

### Transfer Learning

We choose NASNetMobile, ResNet-101 and InceptionResNetV2 as our base models for feature extraction, and then add several layers for classification.

For each training process we first set the trainable attribute of the base model as false and train the model for `50` epochs with a `0.001` base **learning rate** and **batch size** of `64`.
After the first `50` epochs of training, we adjust the trainable attribute of the base model to true and **fine tune** the model for another `30` epochs with the same batch size as the first `50` epochs and `0.0001` **learning rate**.

## Result

The prediction result for the three models for binary classification and multiple classification are shown in the table below. As we can see, for binary classification task, both InceptionResNetV2 and NASNetMobile acheived an accuracy of 100% and the best model for multiple classification is InceptionResNetV2, which acheived an accuracy of 97.45%.

|         Task/Model        | `NASNetMobile` | `InceptionResNetv2` | `ResNet-101` |
| ------------------------- | -------------- | ------------------- | ------------ |
|   Binary Classification   |      100       |         100         |      98      |
|  Multiple Classification  |       95       |        97.45        |      90      |



