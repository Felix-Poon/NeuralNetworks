import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB

"""
CNN-GRU Network to recognise positive and negative reviews from IMDB dataset
"""

class Network(tnn.Module):
    """
    CNN-GRU Network
    Conv -> Relu -> maxpool(size=4) -> Conv -> Relu -> maxpool(size=4) ->
    Conv -> Relu -> global pooling -> GRU -> Linear(64) -> Relu -> linear(1)
    """
    def __init__(self):
        super(Network, self).__init__()
        

        # GRU
        self.gru_layer =  tnn.GRU(50, 100, num_layers = 3, batch_first = True, dropout = 0.2)

        # CNN Layers - generate phrase vectors (NEED RELU)
        self.conv1 = tnn.Conv1d(10,50, kernel_size = 8, padding = 5)
        self.conv2 = tnn.Conv1d(50, 50, kernel_size = 8, padding = 5)
        self.pool = tnn.MaxPool1d(4)

        # GLOBAL POOLING - AdaptiveMaxPool1d(1)
        self.globalPool = tnn.AdaptiveMaxPool1d(1)

        # Output (Linear) layer
        self.fc1 = tnn.Linear(50,1) 

    def forward(self, input, length):
        # GRU
        input = torch.nn.utils.rnn.pack_padded_sequence(input, length, batch_first = True)
        out, hn = self.gru_layer(input)

        # CNN
        input = hn.permute(1,0,2)


        input = self.pool(F.relu(self.conv1(input)))
        input = self.pool(F.relu(self.conv2(input)))
        input = self.globalPool(F.relu(self.conv2(input)))
           
        input = input.permute(0,2,1)
        input = self.fc1(input)
        
        input = torch.squeeze(input)


        return input


class PreProcessing():
    def pre(x):
        # print("PREPROCESSING")
        # print(x)

        # Remove punctuation
        punctuation = ' !"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~ '
        newList = []
        for text in x:
            text = ''.join(ch for ch in text if ch not in punctuation)
            newList.append(text)
        x = newList

        # Stop words - stop_set is taken from NLTK stop-word set.
        stop_set = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", 
        "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "if", "or", "because", "as", "while", "of", 
        "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", 
        "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "each", "few", "more", "most", "other", "some", "such", 
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", ""}

        newList = [word for word in x if word.lower() not in stop_set]
        x = newList

        # remove whitespace
        for text in x:
            text = text.strip()
        
        return x

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        return batch

    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post)


def lossFunc():
    return tnn.BCEWithLogitsLoss()

def main():
    # Use a GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))  
    
    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)
    
    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)


    net = Network().to(device)
    criterion =lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.0005)  # Minimise the loss using the Adam algorithm.


    for epoch in range(10):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

        
    num_correct = 0

    # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print("Saved model")

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")


if __name__ == '__main__':
    main()
