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
        
        self.num_of_classes = len(self.tokenizer)

        # initializing weights:

        self.W = torch.randn((self.num_of_classes, self.num_of_classes), dtype=torch.float, requires_grad=True)

    def train(self, chunk_size:int=5, rate:float=0.1, start:int=0, stop:int=-1):
        stop = len(self.data) if stop==-1 else stop

        # Forward pass 
        neglogloss = torch.tensor(0.0)
        k = 0.0
        for i in range(start, stop-chunk_size):
            for j in range(1, chunk_size):
                inputs = self.data[i:i+j]
                expected = self.data[i+j]

                l0 = F.one_hot(torch.tensor(inputs, dtype=torch.long), num_classes=self.num_of_classes).float() # (j, num_of_classes)
                l0_mean = l0 / l0.sum(dim=1, keepdim=True) # (j, num) / (1, sum) -> (j, num/sum)
                l1= (l0_mean @ self.W) # (j, num_of_classes)
                logits = l1.sum(dim=0) # (num_of_classes)
                counts = logits.exp()
                p = counts / counts.sum()
                neglogloss -= p[expected].log()
                k+=1
        
        loss = neglogloss / k
        
        print(loss)
        
        # Backward pass
        self.W.grad = None
        loss.backward()

        with torch.no_grad():
            self.W -= self.W.grad * rate
    
    def sample(self, inputs:list[int]):
        
        l0 = F.one_hot(torch.tensor(inputs, dtype=torch.long), num_classes=self.num_of_classes).float() # (j, num_of_classes)
        l0_mean = l0 / l0.sum(dim=1, keepdim=True) # (j, num) / (1, sum) -> (j, num/sum)
        l1= (l0_mean @ self.W) # (j, num_of_classes)
        logits = l1.sum(dim=0) # (num_of_classes)
        counts = logits.exp()
        p = counts / counts.sum()
        
        return torch.multinomial(p, 1, replacement=True).tolist()