import torch
import torch.nn.functional as F
from collections.abc import Callable


class Tokenizer:
    def __init__(self, path:str):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        unique_letters_set = set(text)
        unique_letters_set.discard('\n')
        
        unique_letters = list(unique_letters_set)
        unique_letters.sort()
        unique_letters.insert(0, '<empty>')
        
        self.itos = {i:s for i, s in enumerate(unique_letters)}
        self.stoi = {s:i for i, s in enumerate(unique_letters)}
    
    def encode(self, text:str):
        return [self.stoi[c] for c in text]
    
    def decode(self, tokens:list[int]):
        return ''.join([self.itos[i] if i != 0 else '' for i in tokens])
    
    def __len__(self):
        return len(self.itos)

def _base_learning_rate_function(_:float) -> float:
    return 0.1

class NGramNetwork:
    
    def __init__(self, context_size:int, tokenizer:Tokenizer, data_path:str, embedding_vector_size=2, w1_out=32, seed=2147483647):
        
        g = torch.Generator().manual_seed(seed)
        
        self.CONTEXT_SIZE = context_size
        self.TOKENIZER = tokenizer
        
        # --- Reading in data to X (inputs) and Y (outputs)
        
        with open(data_path, 'r') as f:
            words = f.readlines()
        
        X = []
        Y = []
        
        for word in words:
            word = word.rstrip()
            context = [0] * context_size
            for i in tokenizer.encode(word) + [0]:
                X.append(context)
                Y.append(i)
                context = context[1:] + [i]
        
        divider_1 = round(len(X) * 0.8)
        divider_2 = round(len(X) * 0.9)
        
        self.traX,  self.traY  = torch.tensor(X[:divider_1]),          torch.tensor(Y[:divider_1])
        self.devX,  self.devY  = torch.tensor(X[divider_1:divider_2]), torch.tensor(Y[divider_1:divider_2])
        self.testX, self.testY = torch.tensor(X[divider_2:]),          torch.tensor(Y[divider_2:])
        
        
        # --- Creating parameters
        C_IN = len(tokenizer)
        self.EMBEDDING_VECTOR_SIZE = embedding_vector_size
        W1_OUT = w1_out
        W2_OUT = len(tokenizer)
        
        self.C  = torch.randn( ( C_IN,                                      self.EMBEDDING_VECTOR_SIZE ), generator=g )
        self.W1 = torch.randn( ( self.EMBEDDING_VECTOR_SIZE * context_size, W1_OUT                     ), generator=g )
        self.b1 = torch.randn( ( W1_OUT                                                                ))
        
        # Set the last layer to near zero, in order to create a close to uniform distribution in the beginning
        self.W2 = torch.randn( ( W1_OUT,                                    W2_OUT                     ), generator=g) * 0.01
        self.b2 = torch.zeros( ( W2_OUT                                                                ))
        
        self.parameters = [ self.C, self.W1, self.b1, self.W2, self.b2 ]
        
        for p in self.parameters:
            p.requires_grad = True
        
        self.i = 0.
    
    def train(self, batch_size=64, iterations=1000, learning_rate_function:Callable[[float], float]=_base_learning_rate_function, print_out=True, print_out_margin=5, use_dev_set=False):
        for i in range(iterations):
            X, Y = (self.devX, self.devY) if use_dev_set else (self.traX, self.traY)
            
            ix = torch.randint( 0, len(X), (batch_size,) )
           
            emb = self.C[ X[ix] ].view(batch_size, self.CONTEXT_SIZE * self.EMBEDDING_VECTOR_SIZE) # (batch_size, ctx, embed_vec_size) -> (batch_size, ctx*embed_vec_size)
            
            l1 = torch.tanh(emb @ self.W1 + self.b1)
            
            logits = l1 @ self.W2 + self.b2
            
            loss = F.cross_entropy(logits, Y[ix])
            
            for p in self.parameters:
                p.grad = None
            
            loss.backward()
            
            for p in self.parameters:
                p.data -= learning_rate_function(self.i) * p.grad
            
            if print_out:
                if i < print_out_margin or iterations - print_out_margin < i:
                    print(f'Iteration {self.i:>10}: loss={loss.item()}')
                elif i == print_out_margin:
                    print('...')
            
            self.i += 1.
            
    
    def evaluate_test_set(self) -> int:
        with torch.no_grad():
            emb = self.C[ self.testX ].view(-1, self.CONTEXT_SIZE * self.EMBEDDING_VECTOR_SIZE)
            l1 = torch.tanh( emb @ self.W1 + self.b1 )
            
            logits = l1 @ self.W2 + self.b2
            
            loss = F.cross_entropy(logits, self.testY)
            
            return loss.item()
    
    def sample(self, x:list[int]) -> int:
        if len(x) < self.CONTEXT_SIZE:
            x = [0] * (self.CONTEXT_SIZE - len(x)) + x
            
        with torch.no_grad():
            x = torch.tensor(x)
            
            emb = self.C[x].view(self.CONTEXT_SIZE * self.EMBEDDING_VECTOR_SIZE)
            l1 = torch.tanh( emb @ self.W1 + self.b1 )
            logits = l1 @ self.W2 + self.b2
            p = F.softmax(logits, dim=0)
        
            return int(torch.multinomial(p, 1, True).item())

