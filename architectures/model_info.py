from mlp import MultilayerPerceptron
from sklearn.metrics import confusion_matrix

def main():

    model = MultilayerPerceptron()

    for param in model.parameters():
        param.numel()

    print(confusion_matrix(predicted.view(-1), y_test.view(-1)))