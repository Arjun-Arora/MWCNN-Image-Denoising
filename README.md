# MWCNN Image Denoising
Implementation of Multi-level Wavelet-CNN for Image Restoration in Pytorch

Matlab Code: 

https://github.com/lpj0/MWCNN


Citation: 

Liu, Pengju, et al. "Multi-level wavelet-CNN for image restoration." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2018.

#Tutorial
to install correct environment: 
1. conda env create environment.yml
2. conda activate EE367
3. cd ./pytorch_wavelets
4. pip install .
5. cd ../ (return to project root directory)
6. python runMe.py
7. (optional) check out runMe.ipynb with jupyter notebook

Outputs images should be saved to experiments/(model)/images/test_images folder

#Note
If you run into out of memory errors or killed 9, you may want to change the value of "n" in runMe.py to 1000 for each of the args


