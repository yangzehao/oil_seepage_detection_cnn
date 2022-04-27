# oil_seepage_detection_cnn
Using pretrained cnn to detect oil leak through satellite images

## 1. Description of this project:
The detail of the results of this project has been presented in my report (Oil seepage detection report-John (Zehao) Yang.pdf). Briefly, the project solves the oil leak detection problems through analyzing the ocean satellite images. In this project, it involves the analysis of over 200 million image data through convolutional neural network using the pretrained xception model.

## 2. The Platform:
	The OS is MacOS Catalina 10.15.7
	The CPU is 2.4 GHz 8-Core Intel Core i9
	The GPU is AMD Radeon Pro 5500M 8G
	
## 3. The programming language and IDE:
	Python and PyCharm
	
## 4. The description of each fold and file:

(1) The folder 'Comparison of original images, mask images and predicted images' includes the comparison results for test images and the corresponding predicted images;

(2) The folder 'seep_detection' contains the original SAR images and mask images;

(3) The file 'Oil seepage report-John (Zehao) Yang' is the report based on this problem. It shows the detail of the model and results;

(4) The file 'model-hxt(xception)-oil_seepage.h5' is the output DCNN model;

(5) The file 'Oil_Seepage_hxt' is the main module to excute the program. It includes detail code of metrics and image digitalization;

(6) The file 'xception_hxt' is the code Xception pretrained model I use;

(7) The file 'image_comparison' is a function to output and save the comparison results in the folder 'Comparison of original images, mask images and predicted images'.

**The folders ((1),(2)) and the model file ((4)) are shared through the dropbox link considering the large size as below:**
https://www.dropbox.com/scl/fo/2ehymlcrxg3fkyphxcq3t/h?dl=0&rlkey=6tnradum46e3a4d96hlua5l5r





Thanks.

John
