# importing all modules and libraries
import matplotlib.pyplot as plt
import numpy as np
import _pickle as cPickle
from skimage.feature import hog
from PIL import Image, ImageDraw
import cv2


def draw_window(img, x1, y1, x2, y2, width, height, pred, prob):
    # open image and resize to given width and height
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(img, (width, height)) 
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),3)
    cv2.imshow('window', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        capture.release()
        cv2.destroyAllWindows()



def load_data(svc_data):
    # load data from pickle object
    with open(svc_data, 'rb') as fid:
        data = cPickle.load(fid)
        return data


def sliding_window(img, svc_data, pbar):
    print("sliding window...")

    # open image and convert it to gray scale
    image = Image.open(img).convert("L")
    temp_img = Image.open(img)

    # get image dimensions
    width, height = image.size
    print(width, height)

    # load classifier data from svc pickle object
    data = load_data(svc_data)
    clf = data['svc']
    window = data['window']
    pixels_per_cell = data['pixels_per_cell']
    cells_per_block = data['cells_per_block']
    orientations = data['orientations']

    # print(window, orientations, pixels_per_cell, cells_per_block)

    # dimensions of traffic signs during training
    w_width = window
    w_height = window

    # pixels to move in x and y direction
    x_step_size = 25
    y_step_size = 15

    # rows and cols variables to traverse through the image
    rows = width
    cols = height

    # update progress bar
    count = 0
    total = 0

    while rows >= w_width and cols >= w_height:
        x_boxes = ((rows - w_width) // x_step_size) + 1
        y_boxes = ((cols - w_height) // y_step_size) + 1
        total += x_boxes * y_boxes
        rows -= w_width
        cols -= w_height

    rows = width
    cols = height

    # list of predictions for given image
    predictions = []

    # outer while loop to down-sample the image till it remains the size of 1 window(150x150)
    while rows >= w_width and cols >= w_height:
       
        # resize image and set image height to cols and width to rows
        image = image.resize((rows, cols))
        temp_img = temp_img.resize((rows, cols))

        # loop through the entire image
        for y in range(0, cols - (w_height - y_step_size), y_step_size):
            for x in range(0, rows - (w_width - x_step_size), x_step_size):
                # crop a window from the image
                window = image.crop((x, y, x + w_width, y + w_height))

                # get feature descriptor and hog image for the window
                fd, hog_image = hog(window, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                    cells_per_block=cells_per_block, visualize=True, multichannel=False)

                # convert feature descriptor to a numpy array for prediction
                features = np.array([fd])

                # get prediction and probability score for given image window
                pred = clf.predict(features)
                prob = round(np.max(clf.predict_proba(features)), 3)

                if count < total:
                    count += 1
                    val = int((count / total) * 100)
                    pbar.setValue(val)
                    

                # set coordinates for each window
                x1 = x
                y1 = y
                x2 = x + w_width
                y2 = y + w_height

                if round(prob, 2) > 0.5 and pred[0] < 10:
                    # if probability score > 0.55, add prediction to predictions list
                    predictions.append({"Prediction": pred[0], "Prob": prob, "x1": x1, "y1": y1,
                                        "x2": x2, "y2": y2, "height": cols, "width": rows})

                elif round(prob, 2) > 0.5 and int(img.split("/")[-1].split("_")[0]) > 49:
                    predictions.append(
                        {"Prediction": pred[0], "Prob": prob, "x1": x1, "y1": y1,
                         "x2": x2, "y2": y2, "height": cols, "width": rows})

        rows -= w_width
        cols -= w_height


    return predictions
