import cv2
import glob
test_cars = glob.glob('test/*-64.png')

def resize(filename, x=64, y=64):
    img = cv2.imread(filename)
    img2 = cv2.resize(img,(x,y), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(filename[:-4]+"-64.png", img2)


for f in test_cars:
    resize(f)
