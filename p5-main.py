# Useful imports


import glob
import numpy as np
from classifier import Classifier

from moviepy.editor import *
from features import *
# Read in our vehicles and non-vehicles
cars = glob.glob('data/vehicles/**/*.png')
notcars = glob.glob('data/non-vehicles/**/*.png')

import matplotlib.pyplot as plt

import cv2
spatial = 32
histbin = 32
carslen = len(cars)
notcarslen = len(notcars)

cars.extend(notcars)
print(len(cars))
print("Extracting features")
car_features = extract_features(cars, cspace='HSV', spatial_size=(spatial, spatial), hist_bins=histbin, hist_range=(0, 256))

print("Extracted")

X = np.array(car_features)
print(str(X.shape))
# Define a labels vector based on features lists
y = np.concatenate((np.ones(carslen), np.zeros(notcarslen))).flatten()


def crop(filename, xmin, xmax, ymin, ymax, legend):
    img = cv2.imread(filename)
    return img[ymin:ymax, xmin:xmax]


#clf = Classifier.load('p5-hsv.pkl')
clf = Classifier()

clf.train(X, y)

#clf.load('p5.pkl')

clf.save("pk5-hsv.pkl")


def tests_classifier(clf):
    test_cars = glob.glob('test/*-64.png')
    clf_x = extract_features(test_cars, cspace='HSV', spatial_size=(spatial, spatial), hist_bins=histbin, hist_range=(0, 256), load_imgs=True)
    clf_y = np.ones(len(clf_x))

    test_pred = clf.predict(clf_x)

    print("Prediction : " + str(test_pred))
    print("Actual : "+str(clf_y))
    print("Test accuracy = "+ str(np.sum([x==y for x, y in zip(test_pred, clf_y)]) / float(len(clf_y))))

#tests_classifier(clf)

def slide_window(img, x_start_stop, y_start_stop, xy_window, xy_overlap=(0.01, 0.01), output_size=(64,64)):
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    #print(str(xspan)+" / "+str(yspan))
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)

    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Yield image part
            yield (startx, endx, starty, endy, cv2.resize(img[starty:endy, startx:endx], output_size))



def find_cars2(img, clf, window_size):
    output = np.copy(img)
    for (startx, endx, starty, endy, window) in slide_window(img, (700, 1200), (300, 700), window_size):
        features = extract_features([window], cspace='RGB',load_imgs=False)
        prediction = clf.predict(features)
        if prediction == 1:
            print("Car detected")
            cv2.rectangle(output, (startx, starty), (endx, endy), (254, 255, 0), 3)

    return output

# Read in image similar to one shown above
image = cv2.imread('test-image-2.png')

#img2 = find_cars(image, 400, 700, 1, clf, orient=9, pix_per_cell=8, cell_per_block=8, spatial_size=(spatial, spatial), hist_bins=histbin )
img2 = find_cars2(image, clf, (100, 100))
img3 = find_cars2(img2, clf, (250,250))
print(img3)
plt.imshow(img3)
plt.show()



i = 0
def process_image(img):
    ## Define search mask
    img2 = np.copy(img)
    img2 = img2 / 255.0
    global i
    #print(str(i)+ " : " +str(img.shape))
    ## Search sliding windows
    for w in [200, 150, 100, 80, 64]:
        img = find_cars2(img2, clf, (w,w))


    ## Heatmap

    ## Tag image with results
    i += 1

    return img


clip1 = VideoFileClip("project_video.mp4")

white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile("output.mp4", audio=False,fps=25,codec='mpeg4')