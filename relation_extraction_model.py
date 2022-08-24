import torch
import torch.nn as nn
import sklearn.metrics
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class KlueReDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, sentences, labels, max_length=64):
        self.encodings = tokenizer(sentences,
                                   max_length=max_length,
                                   padding='max_length',
                                   truncation=True)
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        
        return item
    
    def __len__(self):
        return len(self.labels)


class AverageMeter():
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def train_epoch(data_loader, model, criterion, optimizer, train=True):
    loss_save = AverageMeter()
    acc_save = AverageMeter()
    
    loop = tqdm(enumerate(data_loader), total=len(data_loader))
    for _, batch in loop:
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'token_type_ids': batch['token_type_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
        }
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(**inputs)
        logits = outputs['logits']
        
        loss = criterion(logits, labels)
        
        if train:
            loss.backward()
            optimizer.step()
            
        preds = torch.argmax(logits, dim=1)
        acc = ((preds == labels).sum().item() / labels.shape[0])
        
        loss_save.update(loss, labels.shape[0])
        acc_save.update(acc, labels.shape[0])
        
    results = {
        'loss': loss_save.avg,
        'acc': acc_save.avg,
    }
    
    return results

def predict(sentence):
    encodings = tokenizer(sentence, 
                          max_length=128, 
                          truncation=True, 
                          padding="max_length", 
                          return_tensors="pt")

    outputs = model(**encodings)

    logits = outputs['logits']

    preds = tag[torch.argmax(logits, dim=1)]
    print(f'sentence: {sentence}\npredict: {preds}')


tag = ["no_relation","org:dissolved","org:founded","org:place_of_headquarters","org:alternate_names","org:member_of","org:members","org:political/religious_affiliation","org:product","org:founded_by","org:top_members/employees","org:number_of_employees/members","per:date_of_birth","per:date_of_death","per:place_of_birth","per:place_of_death","per:place_of_residence","per:origin","per:employee_of","per:schools_attended","per:alternate_names","per:parents","per:children","per:siblings","per:spouse","per:other_family","per:colleagues","per:product","per:religion","per:title"]


# tokenizer = AutoTokenizer.from_pretrained("ainize/klue-bert-base-re")
# model = AutoModelForSequenceClassification.from_pretrained("ainize/klue-bert-base-re")
tokenizer = AutoTokenizer.from_pretrained('ainize/klue-bert-base-re')
model = AutoModelForSequenceClassification.from_pretrained('ainize/klue-bert-base-re')

# # for model train
# df = pd.read_csv("additional_data_for_realtion.csv")
# train_sentences = df["sentence"].values.tolist()
# train_labels = df["label"].values.tolist()

# # For Dataloader
# batch_size = 16

# # For model
# num_labels = 30

# # For train
# learning_rate = 1e-5
# weight_decay = 0.0
# epochs = 1


# entity_special_tokens = {'additional_special_tokens': ['<obj>', '</obj>', '<subj>', '</subj>']}
# num_additional_special_tokens = tokenizer.add_special_tokens(entity_special_tokens)

# train_dataset = KlueReDataset(tokenizer, train_sentences, train_labels)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# model.resize_token_embeddings(len(tokenizer))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        
# # loss function, optimizer 설정
# criterion = nn.CrossEntropyLoss()
# optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# model.to(device)
# for epoch in range(epochs):
#     print(f'< Epoch {epoch+1} / {epochs} >')
    
#     # Train
#     model.train()
    
#     train_results = train_epoch(train_loader, model, criterion, optimizer)
#     train_loss, train_acc = train_results['loss'], train_results['acc']
        
    
#     print(f'train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}')
#     print('=' * 100)

# tokenizer.save_pretrained('./klue-bert-base-re')
# model.save_pretrained('./klue-bert-base-re')



##########################################################################


# Add "<subj>", "</subj>" to both ends of the subject object and "<obj>", "</obj>" to both ends of the object object.
sentence = "<subj>영희</subj>는 아들 <obj>철수</obj>를 사랑한다."
predict(sentence)
