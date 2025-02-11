#Leer los datos
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt','r',encoding="utf-8") as f:
  text = f.read()

#Encoder y decoder 

#Todos los caracteres que aparecen en el texto 
chars = sorted(list(set(text)))
vocab_size = len(chars)
#Mapeado de caracteres a integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s:[stoi[c] if c in stoi else stoi[' '] for c in s] #in -> str out -> int # If the character is not in the vocabulary, replace it with a space
decode = lambda l: ''.join([itos[i] for i in l]) # in -> int out -> str


#Train y test
data = torch.tensor(encode(text),dtype=torch.long)
