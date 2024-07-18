from util import *
import csv
import pandas as pd
import matplotlib.pyplot as plt

file1 = pd.DataFrame(pd.read_csv('200_store_recon_loss.csv'))
file1 = np.array(file1)
file2 = pd.DataFrame(pd.read_csv('500_store_recon_loss.csv'))
file2 = np.array(file2)
y1 = file1[:,0]
y2 = file2[:,0]
print (y1)
print (y2)
x1 = list(range(0, 10001, 250))
x2 = x1
print (x1)

plt.title("Reconstruction loss of 200/500 hidden units in RBM")
plt.plot(x1,y1,label="200nodes",color='r', linewidth=2)
plt.plot(x2,y2,label="500nodes",color='b', linewidth=2)
plt.xlabel("epochs", fontsize=13)
plt.ylabel("Reconstruction loss", fontsize=13)
plt.legend()#
plt.savefig('image.png')  #save picture
plt.show()