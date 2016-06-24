import librosa
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import config as cfg

fig, axs = plt.subplots(2)

X = cPickle.load( open( cfg.fe_mel_fd+ '/a018_90_120.f', 'rb' ) )
axs[0].matshow(np.log(X.T), origin='lower', aspect='auto')
print X.shape

p_y_preds = y = cPickle.load( open( 'Results/p_y_preds.p', 'rb' ) )
axs[1].plot(p_y_preds)
axs[1].axis([0, 129, 0, 1])
axs[1].legend(list(np.arange(15)))
plt.show()