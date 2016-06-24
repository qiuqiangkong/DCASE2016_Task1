import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.ticker as ticker
import config as cfg

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=['a','b']):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # I also added cmap=cmap here, to make use of the 
    # colormap you specify in the function call
    cax = ax.matshow(cm,cmap=cmap)
    #plt.title(title)
    fig.colorbar(cax)
    if labels:
        ax.set_xticklabels([''] + labels, rotation=45, ha='left' )
        ax.set_yticklabels([''] + labels)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.show()
    
cm = np.array([[  59   ,  0  ,   2  ,   0   ,  0   ,  0   ,  0  ,   0    , 0   ,  0   ,  0   ,  0   ,  8   ,  9   ,  0],
     [0    ,55    , 0    , 0    , 0  ,  11 ,    3   ,  0   ,  0   ,  4  ,   0  ,   3    , 0    , 2   ,  0],
     [0    , 0   , 71   ,  0   ,  0  ,   0   ,  0   ,  0    , 0   ,  0    , 0    , 1   ,  0    , 6  ,   0],
     [0    , 1  ,   0   , 67    , 0    , 0   ,  0   ,  0   ,  0    , 0   ,  0  ,  10   ,  0  ,   0 ,    0],
     [0   ,  0   ,  0   ,  0  ,  67  ,   0  ,  0   ,  0   ,  0   ,  0   ,  3   ,  7   ,  0   ,  0   ,  1],
     [0   ,  0    , 0   ,  0     ,0   , 65  ,   0  ,   0   ,  5   ,  8   ,  0   ,  0   ,  0  ,   0    , 0],
     [0    , 1   ,  0   ,  0   ,  2    , 0   , 61   ,  2   ,  8   ,  3   ,  1   ,  0   ,  0 ,    0   ,  0],
     [0   ,  0    , 0   ,  4   ,  0   ,  0    , 0  ,  59    , 0    , 0    , 0    ,10    , 0  ,   2  ,   3],
     [0   ,  0   ,  0  ,   0   ,  0   ,  6   , 13   ,  0  ,  49    , 2    , 2   ,  2   ,  0  ,   4   ,  0],
     [0   ,  4   ,  0   ,  0   ,  0   ,  2   ,  0   ,  0  ,   2   , 70  ,   0   ,  0  ,   0   ,  0   ,  0],
     [0   ,  0   ,  0  ,   0  ,   0   ,  0    , 1   ,  0   ,  0   ,  0  ,  77   ,  0 ,    0  ,   0   ,  0],
     [0   ,  0  ,   0  ,   3   ,  6    , 1    , 0    , 2   ,  1    , 0   ,  0  ,  58,    0   ,  0  ,   7],
     [6    , 7   ,  0    ,10  ,   0,     0 ,    0   ,  1   ,  0    , 0    , 0   ,  1    ,32  ,  21   ,  0],
     [1   ,  7    , 1  ,   0    , 0    , 0    , 0   ,  0 ,    0   ,  0   ,  0    , 0   ,  0   , 69  ,   0],
     [0   ,  1 ,    0   ,  0   ,  4    , 2  ,   0   ,  4   ,  4   ,  0    , 1  ,  27  ,   0   ,  0    ,35]]).astype(float)
print cm.shape
labels = [ 'clearthroat', 'cough', 'doorslam', 'drawer', 'keyboard', 'keysDrop', 'knock' , 'laughter', 'pageturn', 'phone', 'speech' ]
plot_confusion_matrix( cm, labels=cfg.labels )
