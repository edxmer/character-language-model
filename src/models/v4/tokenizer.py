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