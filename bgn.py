import torch
import torch.nn.functional as F

class BigramNetwork:
    def __init__(self, filename, rate=0.1):
        self.rate = rate
        with open(filename, 'r', encoding='utf-8') as f:
            data_raw = f.read()
        
        # Tokenization
        letters_set = set(data_raw)
        letters_set.remove('\n')
        letters = sorted(list(letters_set))


        self.stoi = { s:i for (i, s) in enumerate(letters) }
        self.stoi['<b>'] = max(self.stoi.values())+1
        self.itos = { i:s for (s, i) in self.stoi.items() }

        self.encrypt = lambda xs: [self.stoi[x] for x in xs]
        self.decrypt = lambda xs: [self.itos[x] for x in xs]

        data = ['<b>'] + [ '<b>' if x == '\n' else x for x in data_raw  ] + ['<b>']
        tokenized_data = self.encrypt(data)
    
        xs = torch.tensor(tokenized_data)
        self.ys = torch.tensor(tokenized_data[1:])

        # I want a single input-output layer neural network,
        # with a softmax normalization at the end,
        # then a negative log mean loss.
        self.n = len(self.stoi)
        self.k = len(tokenized_data)

        self.xs = F.one_hot(xs, num_classes=self.n).float() # kxn
        self.W = torch.tensor(torch.randn((self.n, self.n), dtype=torch.float32), requires_grad=True) # nxn

    def train(self):
        
        # Forward pass
        log_counts = self.xs @ self.W # kxn
        counts = log_counts.exp() 
        counts_normalized = counts / counts.sum(dim=1, keepdim=True) # kxn
        # In the sum, i collapsed the 2nd dimension (index 1), and thus got a kx1 tensor because
        # i kept the dimension. Keeping the dimension is necessary, because it ensures that when dividing,
        # pytorch will not push it to the right, and create dimensions on the left.
        # Then i divided counts by it, getting a probability distribution.
        lml = counts_normalized[torch.arange(self.k-1), self.ys].log().mean() 
        nlml = -lml


        # Backward pass
        self.W.grad = None
        nlml.backward()

        # Gradient descent
        with torch.no_grad():
            self.W -= self.W.grad * self.rate

        return nlml

    
    def sample(self, i):
        i = i if isinstance(i, torch.Tensor) else torch.tensor(i)
        xs = F.one_hot(i, num_classes=self.n).reshape((1, self.n)).float()
        log_counts =  xs @ self.W
        counts = log_counts.exp()
        counts_normalized = counts / counts.sum(dim=1, keepdim=True)

        out = torch.multinomial(counts_normalized, num_samples=1, replacement=True)
        return out.data