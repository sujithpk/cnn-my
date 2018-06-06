import numpy as np
import csv
from scipy.fftpack import fft

class DataSet(object):
    def __init__(self, acdata, labels):
        assert acdata.shape[0] == labels.shape[0], (
                "acdata.shape: %s labels.shape: %s" % (acdata.shape,
                                                       labels.shape))
        assert acdata.shape[3] == 1
        acdata = acdata.reshape(acdata.shape[0],
                                    acdata.shape[1] * acdata.shape[2])
        acdata = acdata.astype(np.float32)
        self._acdata = acdata
        self._labels = labels

    @property
    def acdata(self):
        return self._acdata

    @property
    def labels(self):
        return self._labels

# 784*2=1568, 1567*2 + 5 = 3139
with open("/home/sujithpk/Desktop/cnn/train_sig.csv") as file:
    reader=csv.reader(file)
    tr_sig=list(reader) #tr_sig is the list format of csv file train_sig

with open("/home/sujithpk/Desktop/cnn/test_sig.csv") as file:
    reader=csv.reader(file)
    ts_sig=list(reader) #ts_sig is the list format of csv file test_sig

def getData(colno,tr_or_ts):
    # colno = column to be read, tr_or_ts = train data or test data
    ac_sig = np.zeros(3139)
    if tr_or_ts == 11:
        for i in range(3139):
            ac_sig[i] = float(tr_sig[i][colno]) / 2.38
    elif tr_or_ts == 22:
        for i in range(3139):
            ac_sig[i] = float(ts_sig[i][colno]) / 2.38
    
    #sliding window 5 long, step size 2
    ac_smpld = np.zeros(1568)

    for m in range(1568):
        adn = 0.0
        for n in range(5):
            adn = adn + float(ac_sig[m*2 + n]) # sum 
            ac_smpld[m] = adn / 5 #average

    han_wind=np.hanning(1568)
    ac_han=np.multiply(ac_smpld,han_wind)

    #get fft of ac_han
    ac_fft = abs(fft(ac_han))
    ac_data = np.zeros(784) # final result : the training data

    #finding rms of bands
    for i in range(784):
        sq_sum = 0.0
        for j in range(2):
            sq_sum = sq_sum + ac_fft[i*2 + j] * ac_fft[i*2 + j] #squared sum 
            sq_sum = sq_sum /2  #mean of squared sum
            ac_data[i] = np.sqrt(sq_sum) #root of mean of squared sum = rms
    return ac_data

def read_inp(n_train,n_test,one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    print('\n.......Reading data inputs.......\n')
    VALIDATION_SIZE = int(n_train/10) #usually 10% of n_train
    
    train_acdata = np.zeros((n_train,28,28,1))
    for i in range(n_train):
        count=0
        acdat=getData(i,11)
        for j in range(28):
            for k in range(28):
                train_acdata[i,j,k,0]=acdat[count]
                count+=1

    train_labels = np.zeros(n_train)
    n1=int(n_train/9) #28
    cnt1=0
    for i in range(3):
        for j in range(n1):
            for k in range(3):
                train_labels[cnt1] = i*3 +k
                cnt1=cnt1+1

    test_acdata = np.zeros((n_test,28,28,1))
    for i in range(n_test):
        count=0
        acdat=getData(i,22)
        for j in range(28):
            for k in range(28):
                test_acdata[i,j,k,0]=acdat[count] 
                count+=1
  
    ext_lab = np.zeros((n_test,))
    validation_acdata = train_acdata[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_acdata = train_acdata[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]
    data_sets.train = DataSet(train_acdata, train_labels)
    data_sets.validation = DataSet(validation_acdata, validation_labels)
    data_sets.test = DataSet(test_acdata, ext_lab)
    return data_sets
