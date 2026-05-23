import torch
import torch.nn.functional as F


# Note: not finished fully, while this works, it is pretty slow.


class Tokenizer:
    """
    Creates a simple char-by-char tokenizer from a file input.
    """
    def __init__(self, file:str):
        with open(file, "r", encoding='utf-8') as f:
            letters = list(set(list(f.read())))
        letters.sort()
        letters.extend(['<begin>', '<end>'])

        self.stoi = { s:i for (i, s) in enumerate(letters) }
        self.itos = { i:s for (s, i) in self.stoi.items()}
    
    def encode(self, x:list[str]) -> list[int]:
        out = [self.stoi[s] for s in x]
        return out
    
    def decode(self, x:list[int]) -> str:
        out = [self.itos[i] for i in x]
        return "".join(out)
    
    def __len__(self):
        return len(self.stoi)

class NgramNetwork:
    def __init__(self, file:str):
        self.tokenizer = Tokenizer(file)

        self.data = [self.tokenizer.stoi['<begin>']]
        with open(file, "r", encoding='utf-8') as f:
            for char in f.read():
                if char == '\n':
                    self.data.extend(self.tokenizer.encode(['<end>', '<begin>']))
                else:
                    self.data.append(self.tokenizer.stoi[char])
        self.data = self.data[:-1]
        
        self.token_count = len(self.tokenizer)
        self.data_size = len(self.data)
        
        # initializing weights:
        self.W = torch.randn((self.token_count, self.token_count), dtype=torch.float, requires_grad=True) # shape: (token_count, token_count)

    def train(self, chunk_size=5, rate=0.1):
        # Forward pass 
        # 0 : 1
        # 0 : 2
        # 0 : datasize - chunksize
        #  1 : datasize - chunksize + 1
        #   ... 
        #    chunk_size-1 : datasize - chunksize + chunksize
        
        
        
        xs = torch.zeros((self.data_size-chunk_size+1, self.token_count), dtype=torch.float32) # shape: (datasize-1)
        
        nlml = torch.tensor([0], dtype=torch.float)
        k = 0
        
        for i in range(1, self.data_size):
            for j in range(max(0, i-chunk_size), i):
                params = F.one_hot(torch.tensor(self.data[j:i], dtype=torch.long), num_classes=self.token_count).float()
                avg = params.sum(dim=0) / (i-j)
                logits = avg @ self.W # shape: (token_count) * (token_count, token_count) = (token_count)
                counts = logits.exp()
                probabilities = counts / counts.sum()
                nlml += -probabilities[self.data[i]].log()
                k += 1
                
        
        nlml /= k
        print(nlml)
        
        # Backward pass
        self.W.grad = None
        nlml.backward()

        with torch.no_grad():
            self.W -= self.W.grad * rate
    
    def sample(self, inputs:list[int]):
        params = F.one_hot(torch.tensor(inputs, dtype=torch.long), num_classes=self.token_count).float() # (len(inputs), num_of_classes)
        avg = params.sum(dim=0) / len(inputs)
        logits = (avg @ self.W) # (j, num_of_classes)
        counts = logits.exp()
        probabilities = counts / counts.sum()
        
        return torch.multinomial(probabilities, 1, replacement=True).tolist()