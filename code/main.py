import torch
import torch.nn as nn
import torch.optim as optim
import transformers
import json
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import copy
import argparse

import numpy as np
import gc

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# TODO do seeding properly
import random
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

intent_strs = ['Send_digital_object',
           'Get_health_stats',
           'Get_message_content',
           'Add_contact',
           'Initiate_call',
           'Create_note',
           'Add_item_to_list',
           'Create_list',
           'Get_list',
           'Order_menu_item',
           'Find_parking',
           'Get_note',
           'Start_exercise',
           'Stop_exercise',
           'Resume_exercise',
           'Pause_exercise',
           'Log_exercise',
           'Log_nutrition',
           'Check_order_status',
           'Get_bill',
           'Get_security_price',
           'Open_app',
           'Pay_bill',
           'Get_product',
           'Other',
           'Post_message',
           'Record_video',
           'Take_photo',
           'Cancel_ride',
           'Order_ride',
           'BuyEventTickets',
           'Play_game',
           'GetGenericBusinessType']

intent_idxs = { intent: idx for idx, intent in enumerate(intent_strs) }

config = None

def load_jsonl_file(filepath):
    data = []
    with open(filepath, 'r') as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line))
    return data

class RobertaDataset(Dataset):

    def __init__(self, data):
        self.prompts = []
        self.labels = []

        self.test = False
        assert(len(data) > 0)
        if 'output' not in data[0]:
            self.test = True

        for sample in data:
            if sample['history']:
                last_history = sample['history'][-1]
                last_history_str = f"{last_history['user_query']}? {last_history['response_text']}"
            else:
                last_history_str = " "

            # RoBERTa will need to fill this in!
            self.prompts.append(f"[{last_history_str}] {sample['input']}")
            if not self.test:
                gold_intent = sample['output'].split('(')[0].strip()
                label = intent_idxs[gold_intent]
                self.labels.append(label)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        if self.test:
            return (self.prompts[idx],)
        return (self.prompts[idx], self.labels[idx])

class T5Dataset(Dataset):
    
    def __init__(self, data, intents=None):
        self.prompts = []
        self.labels = []

        self.test = False
        if intents is not None:
            self.test = True

        for i in range(len(data)):
            sample = data[i]
            if sample['history']:
                last_history = sample['history'][-1]
                last_history_str = f"{last_history['user_query']}? {last_history['response_text']}"
            else:
                last_history_str = " "

            if self.test:
                intent = intents[i]
            else:
                intent = sample['output'].split('(')[0].strip()
                self.labels.append(sample['output'])

            self.prompts.append(f"[{last_history_str}] {sample['input']} <{intent}>: ")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        if self.test:
            return (self.prompts[idx],)
        return (self.prompts[idx], self.labels[idx])

def dl_collate_fn(batch):
    return [list(a) for a in zip(*batch)]

class T5PrefixTuning(nn.Module):

    def __init__(self, n_prefixes=50):
        super().__init__()

        self.n_prefixes = n_prefixes
        self.prefixes = nn.Parameter(torch.empty((n_prefixes,768)).normal_(0,0.1).to(device))
        
        self.model = transformers.T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

    def forward(self, proc_batch):
        enc_tok_embeds = self.model.shared(proc_batch['input_ids'])
        
        encoder_embeds = torch.cat([
            self.prefixes.unsqueeze(0).expand(proc_batch['input_ids'].size(0),self.n_prefixes,768),
            enc_tok_embeds
        ], dim=1)
        attention_mask = torch.cat([
            torch.ones((proc_batch['input_ids'].size(0), self.n_prefixes)).to(device),
            proc_batch['attention_mask']
        ], dim=1)

        labels = None
        if 'labels' in proc_batch:
            labels = proc_batch['labels']

        return self.model(inputs_embeds=encoder_embeds, attention_mask=attention_mask, labels=labels)

    def generate(self, proc_batch, num_beams=5, max_new_tokens=100):
        enc_tok_embeds = self.model.shared(proc_batch['input_ids'])

        encoder_embeds = torch.cat([
            self.prefixes.unsqueeze(0).expand(proc_batch['input_ids'].size(0),self.n_prefixes,768),
            enc_tok_embeds
        ], dim=1)
        attention_mask = torch.cat([
            torch.ones((proc_batch['input_ids'].size(0), self.n_prefixes)).to(device),
            proc_batch['attention_mask']
        ], dim=1)

        return self.model.generate(
            inputs_embeds=encoder_embeds,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
            
def construct_roberta_forward_fn(model, tokenizer):

    def roberta_forward_fn(batch):
        proc_batch = tokenizer(batch[0], return_tensors="pt", padding=True, truncation=True).to(device)
        
        return model(**proc_batch, labels=torch.tensor(batch[1]).to(device)).loss

    return roberta_forward_fn

def construct_t5_forward_fn(model, tokenizer):

    def t5_forward_fn(batch):
        proc_batch = tokenizer(batch[0], text_target=batch[1], 
                return_tensors="pt", padding=True, truncation=True).to(device)
        
        return model(proc_batch).loss

    return t5_forward_fn

def train(model, forward_fn, train_dl, val_dl, optimizer, scheduler=None, max_epochs=20, patience_lim=2):

    best_model = None
    best_val_loss = 10000
    val_losses = []
    train_losses = []
    patience = 0

    for epoch in range(max_epochs):

        print(f'Epoch {epoch}:')
        train_loss = torch.tensor(0, dtype=torch.float, device=device)
        model.train()
        for batch in tqdm(train_dl):
            optimizer.zero_grad()
            loss = forward_fn(batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.detach()
        
        if scheduler:
            scheduler.step()

        train_loss = train_loss.cpu()
        train_loss /= len(train_dl)
        print(f' Train Loss: {train_loss}')
        train_losses.append(train_loss)

        val_loss = torch.tensor(0, dtype=torch.float, device=device)
        true_labels = []
        pred_labels = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dl):
                loss = forward_fn(batch)
                val_loss += loss.detach()
            
        val_loss = val_loss.cpu()
        val_loss /= len(val_dl)
        val_losses.append(val_loss)

        print(f' Val Loss: {val_loss}')
        print('')

        # early stopping
        if val_loss >= best_val_loss:
            if patience >= patience_lim:
                break
            else:
                patience += 1
        else:
            patience = 0
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model = best_model.cpu()
            print(f'best model: {epoch}')
    
    return best_model, (train_losses, val_losses)

def roberta_generate(model, tokenizer, dl):
    
    labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dl):
            proc_batch = tokenizer(batch[0], return_tensors="pt", padding=True, 
                    truncation=True).to(device)

            outputs = model(**proc_batch)
            labels.append(torch.argmax(outputs.logits, dim=1))
            
    return torch.hstack(labels)

def t5_generate(model, tokenizer, dl):

    gens = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dl):
            proc_batch = tokenizer(batch[0], return_tensors="pt", padding=True, 
                    truncation=True).to(device)

            toks = model.generate(
                proc_batch, 
                num_beams=config.n_beams, 
                max_new_tokens=config.max_new_tokens,
            )

            gens += tokenizer.batch_decode(toks, skip_special_tokens=True)
            
    return gens

def train_roberta(train_data, val_data):

    train_ds = RobertaDataset(train_data)
    val_ds = RobertaDataset(val_data)
    
    train_dl = DataLoader(train_ds, collate_fn=dl_collate_fn, batch_size=config.train_batch_size, num_workers=config.n_workers, shuffle=True)
    val_dl = DataLoader(val_ds, collate_fn=dl_collate_fn, batch_size=config.val_batch_size, num_workers=config.n_workers, shuffle=False)

    model = transformers.RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(intent_strs)).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
    optimizer = optim.Adam(model.parameters(), lr=config.roberta_lr)

    forward_fn = construct_roberta_forward_fn(model, tokenizer)

    best_model, (train_losses, val_losses) = \
        train(model, forward_fn, train_dl, val_dl, optimizer, max_epochs=config.max_roberta_epochs, patience_lim=config.patience)
    torch.save(best_model, config.save_roberta_name)

    del model, tokenizer, optimizer, train_ds, val_ds, train_dl, val_dl, best_model
    gc.collect()

def train_t5(train_data, val_data):

    train_ds = T5Dataset(train_data)
    val_ds = T5Dataset(val_data)
    
    train_dl = DataLoader(train_ds, collate_fn=dl_collate_fn, batch_size=config.train_batch_size, num_workers=config.n_workers, shuffle=True)
    val_dl = DataLoader(val_ds, collate_fn=dl_collate_fn, batch_size=config.val_batch_size, num_workers=config.n_workers, shuffle=False)

    model = T5PrefixTuning(n_prefixes=config.n_prefixes)

    tokenizer = transformers.T5Tokenizer.from_pretrained("t5-base")
    optimizer = optim.AdamW(model.parameters(), lr=config.t5_lr)

    forward_fn = construct_t5_forward_fn(model, tokenizer)

    best_model, (train_losses, val_losses) = \
        train(model, forward_fn, train_dl, val_dl, optimizer, max_epochs=config.max_t5_epochs, patience_lim=config.patience)
    torch.save(best_model, config.save_t5_name)

    del model, tokenizer, optimizer, train_ds, val_ds, train_dl, val_dl, best_model
    gc.collect()

def train_models():

    train_data = load_jsonl_file(config.train_ds_path)
    val_data = load_jsonl_file(config.val_ds_path)

    if config.debug:
        train_data = random.sample(train_data, config.debug_train_len)
        val_data = random.sample(val_data, config.debug_val_len)

    if not config.no_roberta:
        train_roberta(train_data, val_data)

    if not config.no_t5:
        train_t5(train_data, val_data)

def generate_intents(data):
    
    test_ds = RobertaDataset(data)
    test_dl = DataLoader(test_ds, collate_fn=dl_collate_fn, batch_size=config.roberta_batch_size, num_workers=config.n_workers, shuffle=False)

    model = torch.load(config.roberta_path, map_location=device)
    tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")

    intents = roberta_generate(model, tokenizer, test_dl)
    intents = [intent_strs[i] for i in intents]

    del model, tokenizer, test_ds, test_dl
    gc.collect()

    return intents

def test_models():

    test_data = load_jsonl_file(config.test_ds_path)

    intents = generate_intents(test_data)

    test_ds = T5Dataset(test_data, intents=intents)
    test_dl = DataLoader(test_ds, collate_fn=dl_collate_fn, batch_size=config.t5_batch_size, num_workers=config.n_workers, shuffle=False)

    model = torch.load(config.t5_path, map_location=device)
    tokenizer = transformers.T5Tokenizer.from_pretrained("t5-base")

    generations = t5_generate(model, tokenizer, test_dl)

    with open(config.outpath, 'w') as outfile:
        for gen in generations:
            outfile.write(f'{gen.strip()}\n')

    del model, tokenizer, test_ds, test_dl
    gc.collect()

def parse_args():
    
    parser = argparse.ArgumentParser(prog='COL772 A3', 
            description='Intent classification and slot filling model')
    
    subparsers = parser.add_subparsers(dest='command')
    train_parser = subparsers.add_parser('train')
    test_parser = subparsers.add_parser('test')

    train_parser.add_argument('train_ds_path', type=str)
    train_parser.add_argument('val_ds_path', type=str)
    train_parser.add_argument('--no-roberta', action='store_true')
    train_parser.add_argument('--no-t5', action='store_true')
    train_parser.add_argument('--max-t5-epochs', type=int, default=15)
    train_parser.add_argument('--max-roberta-epochs', type=int, default=6)
    train_parser.add_argument('--patience', type=int, default=2)
    train_parser.add_argument('--n-prefixes', type=int, default=20)
    train_parser.add_argument('--n-workers', type=int, default=2)
    train_parser.add_argument('--train-batch-size', type=int, default=16)
    train_parser.add_argument('--val-batch-size', type=int, default=32)
    train_parser.add_argument('--save-t5-name', type=str, default='model_b.pt')
    train_parser.add_argument('--save-roberta-name', type=str, default='model_a.pt')
    train_parser.add_argument('--roberta-lr', type=float, default=1e-5)
    train_parser.add_argument('--t5-lr', type=float, default=5e-5)
    train_parser.add_argument('--debug', action='store_true')
    train_parser.add_argument('--debug-train-len', type=int, default=128)
    train_parser.add_argument('--debug-val-len', type=int, default=32)

    test_parser.add_argument('test_ds_path', type=str)
    test_parser.add_argument('outpath', type=str)
    test_parser.add_argument('--n-beams', type=int, default=5)
    test_parser.add_argument('--t5-path', type=str, default='model_b.pt')
    test_parser.add_argument('--n-workers', type=int, default=2)
    test_parser.add_argument('--roberta-batch-size', type=int, default=32)
    test_parser.add_argument('--t5-batch-size', type=int, default=32)
    test_parser.add_argument('--roberta-path', type=str, default='model_a.pt')
    test_parser.add_argument('--max-new-tokens', type=int, default=128)

    global config
    config = parser.parse_args()
    
if __name__ == "__main__":

    parse_args()

    if config.command == 'train':
        train_models()
    else:
        test_models()
