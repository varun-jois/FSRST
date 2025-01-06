# FSRST
Official repository for the ACCV 2024 paper [Reference-Based Face Super-Resolution Using the Spatial Transformer](https://openaccess.thecvf.com/content/ACCV2024/html/Jois_Reference-Based_Face_Super-Resolution_Using_the_Spatial_Transformer_ACCV_2024_paper.html).


We can train two types of models:
* Models for reference based face super-resolution
* Models for alignment based on the spatial transformer

### Dataprep
##### Reference-based face super-resolution
![The FSRST model](/figs/FSRST.png)
Data needs to be prepared like in the example provided in sample_data/sr. High-resoltion images go in the hr folders for train and validation. Our model uses 3 reference images for each input image which go in the rf folders. The code prepares the low-resolution images from the high-resolution images provided by you. Please ensure that the name for the hr image matches the name of the folder in rf where the references for that image will go. For instance, train/hr/1.png has its references in the folder train/rf/1, with its references named {1,2,3}.png.

##### Reference-based alignment
![The alignment module](/figs/align.png)
Data needs to be prepared like in the example provided in sample_data/align. Here our main image goes in the im folder and the image we are trying to align with an affine transformation goes in the rf folder. That means, the image in the rf folder will be manipulated to align with the image in im. Again, please ensure the names of the main image and the reference match like in  sample_data.

Once you have prepared the datasets, go into configs/sr.yaml and confings/align.yaml and update the train and valid paths.

### Training
To train an SR model simply run the following from the terminal:
```
$ python train.py -c configs/sr.yaml
```
To train an alignment model run:
```
$ python train.py -c configs/align.yaml
```
All models will be saved in the /checkpoints folder. If you are training multiple models remember to change the *name* parameter in the config files otherwise it may just override the previous experiment. Also, if your training crashed in the middle for whatever reason, you can resume from a checkpoint by changing the *epoch_start* parameter in the config. 

### Inference
To perform inference with your saved SR model simply run:
```
$ python inference -c configs/sr.yaml
```
or for alignment
```
$ python inference -c configs/align.yaml
```
These scripts assume that the data you want to do inference on is the validation data that's mentioned in the config files. Change the path if you would like to perform inference on another dataset but make sure its in the format described in the dataprep section. The output will be saved in the /inference folder.

### Citation
Please cite our work if it was helpful in any way. 

@inproceedings{10.1007/978-981-96-0911-6_24,
author = {Jois, Varun Ramesh and DiLillo, Antonella and Storer, James},
title = {Reference-Based Face Super-Resolution Using the&nbsp;Spatial Transformer},
year = {2024},
isbn = {978-981-96-0910-9},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-981-96-0911-6_24},
doi = {10.1007/978-981-96-0911-6_24},
booktitle = {Computer Vision – ACCV 2024: 17th Asian Conference on Computer Vision, Hanoi, Vietnam, December 8–12, 2024, Proceedings, Part IV},
pages = {409–425},
numpages = {17},
keywords = {Reference-Based Super-Resolution, Face Super-Resolution, Image Alignment},
location = {Hanoi, Vietnam}
}

