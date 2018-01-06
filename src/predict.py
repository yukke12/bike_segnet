import numpy as np
import keras
from PIL import Image

from model import SegNet

import dataset

height = 360
width = 480
classes = 2
epochs = 100
batch_size = 1
log_filepath='./logs_100/'

data_shape = 360*480

def writeImage(image, filename):
    """ label data to colored image """
    Sky = [0,0,0]
    Building = [0,255,0]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([Sky, Building])
    for l in range(0,2):
        r[image==l] = label_colours[l,0]
        g[image==l] = label_colours[l,1]
        b[image==l] = label_colours[l,2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    im = Image.fromarray(np.uint8(rgb))
    return im
    # im.save(filename)

def predict(test, n):
    model = keras.models.load_model('weights.' + '{0:02d}'.format(n) + '.hdf5')
    probs = model.predict(test, batch_size=1)

    prob = probs[0].reshape((height, width, classes)).argmax(axis=2)
    return prob

def main():
    model_num = 6
    image_list = ['val0.txt','val1.txt','val3.txt','val9.txt','val11.txt','val13.txt','val18.txt','val22.txt','val28.txt','val32.txt','val34.txt']
    # image_list = ['val0.txt','val10.txt','val11.txt','val12.txt','val13.txt','val15.txt','val16.txt','val18.txt','val1.txt','val21.txt','val22.txt','val3.txt','val5.txt','val8.txt','val9.txt']
    for val in image_list:
        print("loading data...")
        val_num = val.split('.')[0].split('val')[1]
        or_img = '/home/ubuntu/work/segnet/data/bike_scratch/train/' + '{0:04d}'.format(int(val_num))  + '.png' 
        an_img = '/home/ubuntu/work/segnet/data/bike_scratch/trainannot/' + '{0:04d}'.format(int(val_num))  + '.png'
        or_im = Image.open(or_img)
        an_im = Image.open(an_img)

        ds = dataset.DataSet(test_file='val' + val_num + '.txt', classes=classes)
        test_X, test_y = ds.load_data('test') # need to implement, y shape is (None, 360, 480, classes)
        test_X = ds.preprocess_inputs(test_X)
        test_Y = ds.reshape_labels(test_y)
        prob = predict(test_X, model_num)
        result = writeImage(prob, 'val_' + val_num + '.png')
        im = np.concatenate((or_im, an_im, result), axis=1)
        pil_img = Image.fromarray(im)
        pil_img.save('result_'  + '{0:04d}'.format(int(val_num))  + '_' + str(model_num) + '.png')

    """
    for i in range(24, 35):
        prob = predict(test_X, i+1)
        writeImage(prob, 'val' + str(i+1) + '.png')
    prob = predict(test_X)
    writeImage(prob, 'val.png')
    """

if __name__ == '__main__':
    main()

