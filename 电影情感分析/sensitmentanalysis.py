#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torchtext.legacy import data


# In[2]:


SEED=1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic=True

TEXT = data.Field(tokenize='spacy')
LABEL=data.LabelField(dtype=torch.float)


# In[3]:


from torchtext.legacy import datasets
train_data,test_data=datasets.IMDB.splits(TEXT,LABEL)


# In[4]:


print(f'Number of training examples:{len(train_data)}')
print(f'Number of training examples:{len(test_data)}')


# In[5]:


print(vars(train_data.examples[1]))


# In[6]:


import random
train_data,valid_data=train_data.split(random_state=random.seed(SEED))


# 检查每部分有多少数据

# In[7]:


print(f'Number of training examples:{len(train_data)}')
print(f'Number of validation examples:{len(valid_data)}')
print(f'Number of testing examples:{len(test_data)}')


# In[8]:


TEXT.build_vocab(train_data,max_size=25000,vectors="glove.6B.100d",unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)


# In[9]:


print(f"Unique tokens in TEXT vocabulary:{len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary:{len(LABEL .vocab)}")


# In[10]:


print(TEXT.vocab.freqs.most_common(20))


# In[14]:


print(TEXT.vocab.itos[:10])


# In[12]:


PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
print(PAD_IDX)


# In[26]:



sum=0
for i in range(17500):
    sum+=len(vars(train_data.examples[i])['text'])
print(sum/17500)


# In[24]:


print(LABEL.vocab.stoi)


# In[15]:


BATCH_SIZE=64
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator,valid_iterator,test_iterator = data.BucketIterator.splits(
(train_data, valid_data, test_data), 
    batch_size=BATCH_SIZE,
    device=device)


# In[16]:


import torch.nn as nn
import torch.nn.functional as F

class WordAVGModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_dim, output_dim)
    def forward(self, text):
        embedded = self.embedding(text) # [sent len, batch size, emb dim]
        embedded = embedded.permute(1, 0, 2) # [batch size, sent len, emb dim]
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) # [batch size, embedding_dim]
        return self.fc(pooled)


# In[17]:


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
OUTPUT_DIM = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = WordAVGModel(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)


# In[18]:


model


# In[19]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# In[20]:


pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


# In[34]:


import torch.optim as optim

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)


# In[35]:


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc


# In[36]:


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    total_len=0
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()*len(batch.label)
        epoch_acc += acc.item()*len(batch.label)
        total_len+=len(batch.label)
        
    return epoch_loss / total_len, epoch_acc / total_len


# In[37]:


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    total_len=0
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()*len(batch.label)
            epoch_acc += acc.item()*len(batch.label)
            total_len+=len(batch.label)
        
    return epoch_loss / total_len, epoch_acc /total_len


# In[38]:


import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[33]:


N_EPOCHS = 10

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'wordavg-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


# In[41]:


import spacy
nlp = spacy.load('en_core_web_sm')

def predict_sentiment(sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()


# In[42]:


predict_sentiment("This film is terrible")


# In[46]:


predict_sentiment("This film is bad")


# In[47]:


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                           bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text)) #[sent len, batch size, emb dim]
        output, (hidden, cell) = self.rnn(embedded)
        #output = [sent len, batch size, hid dim * num directions]
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)) # [batch size, hid dim * num directions]
        return self.fc(hidden.squeeze(0))


# In[48]:


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, 
            N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)


# In[49]:


print(f'The model has {count_parameters(model):,} trainable parameters')


# In[50]:


model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

print(model.embedding.weight.data)


# In[14]:


optimizer = optim.Adam(model.parameters(),lr=0.1)
model = model.to(device)


# In[53]:


N_EPOCHS = 5
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'lstm-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


# In[ ]:




