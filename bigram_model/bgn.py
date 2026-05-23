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
        letters = sorted(list(letters_set)) + ['<b>']


        # Create token to int map, and its inverse
        self.stoi = { s:i for (i, s) in enumerate(letters) }
        self.itos = { i:s for (s, i) in self.stoi.items() }

        # Create functions for encrypting and decrypting tokens
        self.encrypt = lambda xs: [self.stoi[x] for x in xs]
        self.decrypt = lambda xs: [self.itos[x] for x in xs]

        # Tokenize the training data
        data = ['<b>'] + [ '<b>' if x == '\n' else x for x in data_raw  ] + ['<b>']
        tokenized_data = self.encrypt(data)

        
        xs = torch.tensor(tokenized_data)
        self.ys = torch.tensor(tokenized_data[1:]) # this is the expected output

        # I want a single input-output layer neural network,
        # with a softmax normalization at the end,
        # then a negative log mean loss.
        
        self.n = len(self.stoi) # n: token size
        self.k = len(tokenized_data) # k: training data size

        self.xs = F.one_hot(xs, num_classes=self.n).float() # shape: (k, n), this is the input data turned into one-hot vectors
        self.W = torch.randn((self.n, self.n), dtype=torch.float32, requires_grad=True) # shape: (n, n), these are the weights, 

    def train(self):
        
        # Forward pass
        log_counts = self.xs @ self.W # shape: (k, n)
        counts = log_counts.exp() # exponentiating it to make everything positive
        counts_normalized = counts / counts.sum(dim=1, keepdim=True) # shape: (k, n)
        
        # In the sum, i collapsed the 2nd dimension (index 1), and thus got a (k, 1) tensor because
        # I kept the dimension. Keeping the dimension is necessary, because it ensures that when dividing,
        # pytorch will not push it to the right, and create dimensions on the left.
        # Then I divided counts by it, getting a probability distribution for each input token (row).
        
        nlml = -counts_normalized[torch.arange(self.k-1), self.ys].log().mean()
        # Now I take the probability assigned to the correct output from each row, take their log and average them.
        # This produces a good loss functions, because we want the model to be confidently correct ->
        # so we want a likelihood as close to 1 as possible. But taking a product of all these numbers
        # would be very ugly, slow and inaccurate. So, we use the properties of a logarithm to be able to
        # use summations instead of products. But because all probabilities are between 0 and 1, their logs
        # will be negative numbers as well, and the closer they are to 1, the closer their logs would be to
        # 0. So we need to *(-1) all of them to get a number that is positive, and better for us the closer
        # it is to zero, which is exactly what a loss function needs to be. The mean is taken to decouple
        # the magnitude of the gradients from the training batch size.


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