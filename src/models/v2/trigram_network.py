import torch
import torch.nn.functional as F

class Tokenizer:
    """
    Creates a simple char-by-char tokenizer from a file input.
    """
    def __init__(self, file:str):
        with open(file, "r", encoding='utf-8') as f:
            letters = list(set(list(f.read())))
        letters.sort()
        letters.append('<b>')

        self.stoi = { s:i for (i, s) in enumerate(letters) }
        self.itos = { i:s for (s, i) in self.stoi.items()}
    
    def encode(self, x:list[str]) -> list[int]:
        out = []
        for s in x:
            out.append(self.stoi[s])
        return out
    
    def decode(self, x:list[int]) -> str:
        out = []
        for i in x:
            out.append(self.itos[i])
        return "".join(out)
    
    def __len__(self):
        return len(self.stoi)

        
        


class TrigramNetwork:
    def __init__(self, file:str, rate=0.1):
        self.file = file
        self.rate = rate        

        # Tokenizer:
        self.tokenizer = Tokenizer(file)
        self.k = len(self.tokenizer)
        
        # Create input-output triplets
        tokenized = self.tokenize()
        
        x1s = tokenized[:-2]
        x2s = tokenized[1:-1]
        ys = tokenized[2:]

        self.n = len(x1s)


        x1s = torch.tensor(x1s, dtype=torch.long)
        x2s = torch.tensor(x2s, dtype=torch.long)

        self.x1s = F.one_hot(x1s, num_classes=self.k).float() # input's first characters
        self.x2s = F.one_hot(x2s, num_classes=self.k).float() # input's second characters

        self.ys = torch.tensor(ys, dtype=torch.long)

        # Initialize weights

        # (n, k)   *   (k, k)  ->  (n, k)
        # (n, k)   +   (1, k)  ->  (n, k)
    
        self.W = torch.randn((self.k, self.k), dtype=torch.float32, requires_grad=True)
    
    def tokenize(self):
        tokenized = []
        with open(self.file, "r", encoding="utf-8") as f:
            for char in f.read():
                if char == '\n':
                    tokenized.extend(self.tokenizer.encode(['<b>', '<b>']))
                else:
                    tokenized.append(self.tokenizer.stoi[char])
        return tokenized
    
    def train(self):
        
        # Forward pass
        logits = ((self.x1s + self.x2s)/2) @ self.W
        counts = logits.exp()
        probabilities = counts / counts.sum(dim=1, keepdim=True) # (n, k)
        
        nlml = -probabilities[torch.arange(self.n), self.ys].log().mean()

        
        print("loss:", nlml)

        # Backward pass

        self.W.grad = None
        nlml.backward()

        with torch.no_grad():
            self.W -= self.W.grad * self.rate


    def sample(self, i1, i2):
        i1 = i1 if isinstance(i1, torch.Tensor) else torch.tensor(i1)
        i2 = i2 if isinstance(i2, torch.Tensor) else torch.tensor(i2)

        x1s = F.one_hot(i1, num_classes=self.k).float()
        x2s = F.one_hot(i2, num_classes=self.k).float()

        logits = (x1s @ self.W) + (x2s @ self.W)
        counts = logits.exp()
        p = counts / counts.sum()

        return torch.multinomial(p, num_samples=1, replacement=True)

        

def __init__():
    pass