# MWCNN Image Denoising
Implementation of Multi-level Wavelet-CNN for Image Restoration in Pytorch

![image](./Project_Report_Files/MWCNN_Poster.png)

## Matlab Code: 

https://github.com/lpj0/MWCNN


Citation: 

Liu, Pengju, et al. "Multi-level wavelet-CNN for image restoration." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2018.

# Tutorial
to install correct environment: 
1. conda env create environment.yml
2. conda activate EE367
3. cd ./pytorch_wavelets
4. pip install .
5. cd ../ (return to project root directory)
6. python runMe.py
7. (optional) check out runMe.ipynb with jupyter notebook

Outputs images should be saved to experiments/(model)/images/test_images folder

# Note
If you run into out of memory errors or killed 9, you may want to change the value of "n" in runMe.py to 1000 for each of the args

## TODO
1. Fix dataloading code so that we don't overwhelm system memory to write patches
2. Don't load patches to system memory, just load filenames and then read them as queried by DataLoader
3. Retrain on entire dataset and run for longer than 6 hours

