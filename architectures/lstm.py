import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LSTM(nn.Module):
    
    def __init__(self, input_size = 1, hidden_size = 50, out_size = 1):
        super().__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)
        # Hidden state, Cell state
        self.hidden = (torch.zeros(1,1,hidden_size), torch.zeros(1,1,hidden_size))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1))

        pred = self.linear(lstm_out.view(len(seq),-1))
        return pred[-1] # pred will output everything in the sequence

x = torch.linspace(0, 799, 800)
y = torch.sin(x * 2 * 3.1416 / 40)

test_size = 40

train_set = y[:-test_size]
test_set = y[-test_size:]

def input_data(seq, window_size):
    out = []

    num_data_pts = seq.size()[0]

    for i in range(num_data_pts - window_size):
        data_point = seq[i:i + window_size]
        label = seq[i + window_size:i + window_size + 1]
        out.append((data_point, label))

    return out

def main():
    out = input_data(train_set, 40)
    print(out[0])
    print(out[1])
    print(len(out))

    torch.manual_seed(42)

    model = LSTM()

    criterion = nn.MSELoss()
    optimizer  = torch.optim.SGD(model.parameters(), lr = 0.01)

    window_size = 40
    train_data = input_data(train_set, window_size)
    test_data = input_data(test_set, window_size)


    epochs = 10
    future = 40

    for i in range(epochs):
        for seq, y_train in train_data:
            
            # reset the parameters and hidden states
            optimizer.zero_grad()
            model.hidden = (torch.zeros(1,1,model.hidden_size),
                            torch.zeros(1,1,model.hidden_size))
            
            y_pred = model(seq)
            
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            
        print(f'Epoch: {i+1:2} Loss: {loss.item():10.8f}')
        
        # MAKE PREDICTIONS
        # start with a list of the last 10 training records
        preds = train_set[-window_size:].tolist()

        for f in range(future):  
            seq = torch.FloatTensor(preds[-window_size:])
            with torch.no_grad():
                model.hidden = (torch.zeros(1,1,model.hidden_size),
                                torch.zeros(1,1,model.hidden_size))
                preds.append(model(seq).item())
                
        loss = criterion(torch.tensor(preds[-window_size:]),y[760:])
        print(f'Loss on test predictions: {loss}')

        # Plot from point 700 to the end
        plt.figure(figsize=(12,4))
        plt.xlim(700,801)
        plt.grid(True)
        plt.plot(y.numpy())
        plt.plot(range(760,800),preds[window_size:])
        plt.show()
if __name__ == '__main__':
    main()