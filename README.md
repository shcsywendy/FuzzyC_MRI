# FuzzyC_MRI
ENHANCING FUZZY CLUSTERING METHODS FOR IMAGE SEGMENTATION USING SPATIAL INFORMATION
## Project Description
Image segmentation is a very challenging task for image understanding because of the diverse and complex nature of images. Researchers have proposed and continue to propose different algorithms based on the fuzzy clustering method, in particular Fuzzy C- Means (FCM) and its modifications. Fuzzy clustering allows an object to belong to all clusters with a specified membership degree and is uniquely effective when boundaries between clusters of data are ambiguous. This thesis investigates an open competition model combining local and global spatial information for FCM. The basic principle is that pixels in boundary areas have the possibility of belonging to nearby sub-area. This model uses FCM for initial clustering and then again in establishing the influence competition procedure. The open competition model is evaluated using MRI brain images from BrainWeb Experimenting with various model parameters has improved the accuracy rate over traditional FCM and other improvements using spatial information.
## Data Resources
https://brainweb.bic.mni.mcgill.ca/brainweb/
## Thesis link:
https://etd.ohiolink.edu/acprod/odb_etd/etd/r/1501/10?clear=10&p10_accession_num=miami1556555486273


## Requirements
Python 3.x
NumPy
Matplotlib
OpenCV
scikit-fuzzy
PIL
Setup
Ensure you have all required libraries installed. You can install them using pip:

pip install numpy matplotlib opencv-python scikit-fuzzy pillow
## Usage
Reading and Processing Images: The script reads raw binary image data, processes it using fuzzy clustering, and visualizes the results.
## Functions:
readImageFromRawb(path): Reads an image from a raw binary file.
change_color_fuzzycmeans(cluster_membership, clusters): Applies Fuzzy C-Means clustering to the image data.
getColorImage(u, cntr, rows, cols): Generates a color-coded image based on cluster membership.



