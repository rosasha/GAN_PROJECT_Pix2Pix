# GAN_PROJECT_Pix2Pix
Project for DLS

Реализация преобразования изображения в изображение (pix2pix) с помощью условных состязательных сетей и библиотеки PyTorch

## Maps dataset
* Image is resized to 256x256 image (Original size: 600x600)
* Number of training images: 1,096
* Number of test images: 1,098
### Results
* Adam optimizer is used. Learning rate = 0.0002, batch size = 1, # of epochs = 100 (из-за ограниченности по времени, для полной имплементации требуется 200 эпох)
* Generated images using test data

    |1st column: Input / 2nd column: Generated / 3rd column: Target|
    |:---:|
    |![](image/maps/label_1.png)|
    |![](maps_test_results/Test_result_560.png)|
    |![](maps_test_results/Test_result_627.png)|
    |![](maps_test_results/Test_result_746.png)|

