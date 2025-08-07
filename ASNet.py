import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# 加载数据
npzfile_multi = np.load('.npz')
M_1 = npzfile_multi['arr1']
M_2 = npzfile_multi['arr2']
M_3 = npzfile_multi['arr3']
l_1 = npzfile_multi['arr4']
bvec_1 = npzfile_multi['arr5']

len_size = int(len(l_1) * 1)
M31 = M_1[:len_size, :, :, :] 
M32 = M_2[:len_size, :, :, :] 
M33 = M_3[:len_size, :] 
bvec3 = bvec_1[:len_size, :, :]
l3 = l_1[:len_size]

M1_train1, M1_Test, M2_train1, M2_Test, M3_train1, M3_Test, B_train1, B_Test, l_train1, l_Test = \
    train_test_split(M31, M32, M33, bvec3, l3, test_size=0.1, random_state=42)
    
M1_train, M1_test, M2_train, M2_test, M3_train, M3_test, B_train, B_test, l_train, l_test = \
    train_test_split(M1_train1, M2_train1, M3_train1, B_train1, l_train1, test_size=0.2, random_state=42)

train_data = TensorDataset(
    torch.tensor(M1_train, dtype=torch.float32),
    torch.tensor(M2_train, dtype=torch.float32),
    torch.tensor(M3_train, dtype=torch.float32),
    torch.tensor(B_train, dtype=torch.float32),
    torch.tensor(l_train, dtype=torch.long)
)

test_data = TensorDataset(
    torch.tensor(M1_test, dtype=torch.float32),
    torch.tensor(M2_test, dtype=torch.float32),
    torch.tensor(M3_test, dtype=torch.float32),
    torch.tensor(B_test, dtype=torch.float32),
    torch.tensor(l_test, dtype=torch.long)
)

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

class CNNModel(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv1a = nn.Conv2d(input_channels, 32, (3, 8), padding='same')
        self.conv1b = nn.Conv2d(input_channels, 32, (8, 3), padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 32, (3, 3), padding='same') 
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding='same')
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 32)  
            
    def forward(self, x):
        x1 = F.relu(self.conv1a(x))
        x2 = F.relu(self.conv1b(x))
        x = torch.cat([x1, x2], dim=1)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        return x


class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
            nn.Linear(1, 1),
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        attn_weights = self.attn(out)
        attn_weights = self.sigmoid(attn_weights.squeeze(-1))
        weighted = out * attn_weights.unsqueeze(-1)
        return weighted.sum(dim=1)

class CoAttention(nn.Module):
    def __init__(self, dim1, dim2, dim3, hidden_dim=128, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.proj1 = nn.Linear(dim1, hidden_dim)
        self.proj2 = nn.Linear(dim2, hidden_dim)
        self.proj3 = nn.Linear(dim3, hidden_dim)
        
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.proj_back1 = nn.Linear(hidden_dim, dim1)
        self.proj_back2 = nn.Linear(hidden_dim, dim2)
        self.proj_back3 = nn.Linear(hidden_dim, dim3)
        
        self.norm1 = nn.LayerNorm(dim1)
        self.norm2 = nn.LayerNorm(dim2)
        self.norm3 = nn.LayerNorm(dim3)

    def forward(self, x1, x2, x3):
        x1_proj = self.proj1(x1).unsqueeze(1)  # [batch, 1, hidden_dim]
        x2_proj = self.proj2(x2).unsqueeze(1)  # [batch, 1, hidden_dim]
        x3_proj = self.proj3(x3).unsqueeze(1)  # [batch, 1, hidden_dim]
        
        combined = torch.cat([x1_proj, x2_proj, x3_proj], dim=1) 
        
        attn_output, _ = self.multihead_attn(combined, combined, combined)
        
        out1 = attn_output[:, 0, :]  
        out2 = attn_output[:, 1, :]
        out3 = attn_output[:, 2, :]
        
        out1 = self.proj_back1(out1)
        out2 = self.proj_back2(out2)
        out3 = self.proj_back3(out3)
        
        out1 = self.norm1(x1 + out1)
        out2 = self.norm2(x2 + out2)
        out3 = self.norm3(x3 + out3)
        
        return out1, out2, out3

class SpeechEmotionModel(nn.Module):
    def __init__(self, mfcc_shape, max_length=64):
        super().__init__()
        self.max_length = max_length
        
        in_channels_mfcc = mfcc_shape[-1] if len(mfcc_shape) == 4 else 1
        self.mfcc_cnn = CNNModel(in_channels_mfcc)
        self.fbank_cnn = CNNModel(in_channels_mfcc)
        
        self.attention1 = AttentionModel(32, 32)  
        self.attention2 = AttentionModel(32, 32)  
        self.attention3 = AttentionModel(314, 314) 
        
        self.co_attention = CoAttention(
            dim1=32,  
            dim2=32,  
            dim3=314, 
            hidden_dim=128,  
            num_heads=4     
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32+32+314+768, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Softmax(dim=1)
        )
        
        self.target_layers = {
            'sp_1': None, 'sp_2': None, 'sp_3': None, 'sp_4': None
        }  

    def forward(self, mfcc, fbank, low_level, bert):
        # MFCC
        mfcc_feat = self.mfcc_cnn(mfcc.permute(0, 3, 1, 2)) 
        mfcc_attn = self.attention1(mfcc_feat)
        self.target_layers['sp_1'] = mfcc_attn

        # FBank
        fbank_feat = self.fbank_cnn(fbank.permute(0, 3, 1, 2))
        fbank_attn = self.attention2(fbank_feat)
        self.target_layers['sp_2'] = fbank_attn

        # Low-level
        low_attn = self.attention3(low_level)
        self.target_layers['sp_3'] = low_attn

        # BERT
        bert_cls = bert[:, 0, :]  
        self.target_layers['sp_4'] = bert_cls
        
        mfcc_co, fbank_co, low_co = self.co_attention(mfcc_attn, fbank_attn, low_attn)
        combined = torch.cat([mfcc_co, fbank_co, low_co, bert_cls], dim=1)
        return self.classifier(combined)

# 初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mfcc_shape = M1_train.shape
model = SpeechEmotionModel(mfcc_shape[1:]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

def train_epoch(model, dataloader):
    model.train()
    total_loss, correct, total = 0, 0, 0

    activations = {f'sp_{i}': [] for i in range(1, 5)}
    all_labels = []
    
    for mfcc, fbank, low, bert, labels in dataloader:
        inputs = [t.to(device) for t in (mfcc, fbank, low, bert)]
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(*inputs)
        
        for k, v in model.target_layers.items():
            activations[k].append(v.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += len(labels)
    
    collected_activations = {}
    for k, v_list in activations.items():
        collected_activations[k] = np.concatenate(v_list)
    all_labels = np.concatenate(all_labels)
    
    return total_loss/len(dataloader), correct/total, collected_activations, all_labels

def evaluate(model, dataloader, collect_activations=False):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    if collect_activations:
        activations = {f'sp_{i}': [] for i in range(1, 5)}
        all_labels = []
    
    with torch.no_grad():
        for mfcc, fbank, low, bert, labels in dataloader:
            inputs = [t.to(device) for t in (mfcc, fbank, low, bert)]
            labels = labels.to(device)
            
            outputs = model(*inputs)
            
            if collect_activations:
                for k, v in model.target_layers.items():
                    activations[k].append(v.detach().cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += len(labels)
    
    # 返回结果
    if collect_activations:
        collected_activations = {}
        for k, v_list in activations.items():
            collected_activations[k] = np.concatenate(v_list)
        all_labels = np.concatenate(all_labels)
        return total_loss/len(dataloader), correct/total, collected_activations, all_labels
    return total_loss/len(dataloader), correct/total

# 训练
epochs = 20
for epoch in range(epochs):
    train_loss, train_acc, _, _ = train_epoch(model, train_loader)
    test_loss, test_acc = evaluate(model, test_loader)
    print(f'Epoch {epoch + 1}/{epochs}:')
    print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')