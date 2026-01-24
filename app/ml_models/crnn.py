import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.embedding = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        # Simplified CNN mirroring the high-performing notebook logic
        # Input: (1, 50, 200)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # (32, 25, 100)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # (64, 12, 50)
        )
        
        # Mapping layer (Bottleneck/Refinement)
        # Height is 12, channels 64 -> 12 * 64 = 768
        self.map_to_rnn = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )
        
        # RNN Layers
        # Input dim is 64 from map_to_rnn
        # Using LSTM layers with dropout as per notebook
        self.rnn = nn.Sequential(
            BidirectionalLSTM(64, 128, 256),
            nn.Dropout(0.25),
            BidirectionalLSTM(256, 64, num_classes)
        )
        
    def forward(self, x):
        # x: [batch, 1, 50, 200]
        conv = self.cnn(x)
        b, c, h, w = conv.size() 
        
        # Reshape for Linear layer: [batch, width, channels * height]
        # w is 50 (sequence length)
        conv = conv.view(b, c * h, w)
        conv = conv.permute(0, 2, 1) # [batch, 50, 768]
        
        # Map to RNN expected dimension
        rnn_input = self.map_to_rnn(conv) # [batch, 50, 64]
        
        # Permute for PyTorch RNN: [seq_len, batch, input_size]
        rnn_input = rnn_input.permute(1, 0, 2) # [50, batch, 64]
        
        output = self.rnn(rnn_input)
        return torch.nn.functional.log_softmax(output, dim=2)
