import pickle


with open('checkpoints/CIFAR10cross_entropy.history', "rb") as file_pi:
    history = pickle.load(file_pi)
    print(history)