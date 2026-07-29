import numpy as np
import matplotlib.pyplot as plt

train_loss_list=[]
cnt=0
step=500
with open("train_loss.txt", "r",encoding="utf-8") as f:
    for line in f:
        if cnt%step==0:
            train_loss_list.append(float(line))

        cnt+=1

plt.xlabel("batch")
plt.ylabel("loss")
plt.plot(train_loss_list,label="train_loss")
plt.legend()
plt.grid(alpha=0.3)
plt.show()