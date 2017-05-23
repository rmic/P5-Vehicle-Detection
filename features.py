import numpy as np
from skimage.feature import hog
import cv2
from matplotlib import image as mpimg
# HOG
# Adapted from code provided in the course
def get_hog_features(img, orient, pix_per_cell, cell_per_block):

    hog_features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=False, feature_vector=False)
    return hog_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='HSV', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=8, load_imgs=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for f in imgs:
        # Read in each one by one
        if(load_imgs):
            image = cv2.imread(f)
        else:
            image = f
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        else: feature_image = np.copy(image)
        # Apply bin_spatial() to get spatial color features
        ch1 = feature_image[:, :, 0]
        ch2 = feature_image[:, :, 1]
        ch3 = feature_image[:, :, 2]

        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block).ravel()
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block).ravel()
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block).ravel()
        hog_features = np.hstack((hog1, hog2, hog3))

        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Append the new feature vector to the features list
        stack = np.hstack((spatial_features, hist_features, hog_features))

        features.append(stack)
    # Return list of feature vectors
    return features



# Define a function to compute color histogram features
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return np.array(features).astype(np.float64)


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return  hist_features