import pickle

from matplotlib import pyplot as plt


with open('checkpoints/CIFAR10cross_entropy.history', "rb") as file_pi:
    sd_history = pickle.load(file_pi)
    
with open('checkpoints/CIFAR10fusion_cross_entropy.history', "rb") as file_pi:
    fu_history = pickle.load(file_pi)
    
with open('checkpoints/CIFAR10symmetrical_cross_entropy.history', "rb") as file_pi:
    sy_history = pickle.load(file_pi)

epoch=[i for i in range(0,200)]

# plt.plot(epoch, sd_history['accuracy'],label='condition1_acc')
# plt.plot(epoch, sd_history['val_accuracy'], label='condition1_val_acc')

# plt.plot(epoch, fu_history['accuracy'],label='condition2_acc')
# plt.plot(epoch, fu_history['val_accuracy'], label='condition2_val_acc')

# plt.plot(epoch, sy_history['accuracy'],label='condition3_acc')
# plt.plot(epoch, sy_history['val_accuracy'], label='condition3_val_acc')

plt.plot(epoch, sd_history['loss'],label='condition1_loss')
plt.plot(epoch, fu_history['loss'],label='condition2_loss')
plt.plot(epoch, sy_history['loss'],label='condition3_loss')




plt.legend()
plt.savefig("pic/loss")
