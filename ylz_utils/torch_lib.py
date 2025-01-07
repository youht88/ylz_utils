import math
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np

class TransformerModel(nn.Module):
    def __init__(self, *, source_dim:int, d_model:int, nhead:int=4, n_layers:int=6,dropout=0.1,target_dim:int=1,is_embedded:bool=False,is_positional:bool=False,mapping=None):
        super(TransformerModel, self).__init__()
        self.mapping=mapping
        self.is_embedded = is_embedded
        self.is_positional = is_positional
        self.paramers = {"source_dim":source_dim,"d_model":d_model,"target_dim":target_dim,
                         "nhead":nhead,"n_layers":n_layers,"dropout":dropout,
                         "is_embedded":is_embedded,"is_positional":is_positional}
        self.source_embedder = nn.Linear(source_dim, d_model)
        self.target_embedder = nn.Linear(target_dim, d_model)
        self.positional_train = PositionalEncoding(d_model,dropout=dropout)
        self.positional_eval = PositionalEncoding(d_model,dropout=0)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=n_layers,num_decoder_layers=n_layers,dropout=dropout)
        self.fc = nn.Linear(d_model, target_dim)

    def forward(self, src, tgt):
        if not self.is_embedded:
            src = self.source_embedder(src)
            tgt = self.target_embedder(tgt)
        if not self.is_positional:
            if self.training:
                src = self.positional_train(src)
                tgt = self.positional_train(tgt)
            else:
                src = self.positional_eval(src)
                tgt = self.positional_eval(tgt)

        output = self.transformer(src,tgt)
        output = self.fc(output)
        return output

class TorchLib:
    @classmethod
    def to_df(cls,tensor:torch.Tensor,columns=[])->pd.DataFrame:
        numpy_array = tensor.numpy()
        if columns:
            df = pd.DataFrame(numpy_array,columns=columns)
        else:
            df = pd.DataFrame(numpy_array)
        return df
    @classmethod
    def encode_mapping(cls,df:pd.DataFrame,source_column,target_column,key_column)->torch.Tensor:
        source_mapping_records = {}
        target_mapping_records = {}
        print("old df length=",len(df))
        df = df[source_column+target_column+[key_column]].dropna()
        print("no na df length=",len(df))
        df=df.copy()
        columns = df.columns
        for col in columns:
            if col==key_column:
                continue
            if col in source_column: 
                source_mapping_dict = {}
                if df[col].dtypes == 'object':
                    unique_values = df[col].unique()
                    for index, value in enumerate(unique_values):
                        source_mapping_dict[value] = index
                    # 使用map函数根据映射字典进行替换
                    df[col] = df[col].map(source_mapping_dict)
                # 每列标准化
                mean = df[col].mean()
                std = df[col].std()
                df[col] = (df[col] - mean)/std
                source_mapping_records[col] = {
                    'positions': df.columns.get_loc(col),
                    'mean':mean,'std':std,
                    'mapping': source_mapping_dict
                }
                
            if col in target_column:
                if col in source_column:
                    #col已经被编码，可以直接使用
                    target_mapping_records[col] = source_mapping_records[col]
                    continue
                target_mapping_dict = {}
                if df[col].dtypes == 'object':
                    unique_values = df[col].unique()
                    for index, value in enumerate(unique_values):
                        target_mapping_dict[value] = index
                    # 使用map函数根据映射字典进行替换
                    df[col] = df[col].map(target_mapping_dict)
                # 每列标准化
                mean = df[col].mean()
                std = df[col].std()
                df[col] = (df[col] - mean)/std               
                target_mapping_records[col] = {
                    'positions': df.columns.get_loc(col),
                    'mean':mean,'std':std,
                    'mapping': target_mapping_dict
                } 
        return (df,source_mapping_records,target_mapping_records)

    @classmethod
    def get_dataloader(cls,df:pd.DataFrame,source_column:list[str],target_column:list[str],key_column:str,
                       source_seq:int,target_seq:int=1,split:float=0.8,batch_size:int=32,filter:list[str]=[])->tuple[DataLoader,DataLoader,dict,dict]:
        assert(source_column and target_column)
        assert(source_seq and target_seq)
        df,source_mapping,target_mapping = cls.encode_mapping(df,source_column,target_column,key_column)
        data_len = len(df) - source_seq - target_seq + 1
        # source_seq_data = torch.stack([source_data[i:i+source_seq] for i in range(data_len)])
        # target_seq_data = torch.stack([target_data[i+source_seq:i+source_seq+target_seq] for i in range(data_len)])
        if filter:
            source_seq_data = torch.stack([torch.from_numpy(df[i:i+source_seq][source_column].values).float() 
                                    for i in range(data_len) if eval(f"{key_column} in {filter}",{key_column:df[i:i+source_seq][key_column].iloc[-1]})])
            target_seq_data = torch.stack([torch.from_numpy(df[i+source_seq:i+source_seq+target_seq][target_column].values).float() 
                                    for i in range(data_len) if eval(f"{key_column} in {filter}",{key_column:df[i:i+source_seq]['date'].iloc[-1]})])
        else:
            source_seq_data = torch.stack([torch.from_numpy(df[i:i+source_seq][source_column].values).float() 
                                    for i in range(data_len)])
            target_seq_data = torch.stack([torch.from_numpy(df[i+source_seq:i+source_seq+target_seq][target_column].values).float() 
                                    for i in range(data_len)])

        train_size = int(len(target_seq_data)*split)        
        x_train,x_test = source_seq_data[:train_size],source_seq_data[train_size:]
        y_train,y_test = target_seq_data[:train_size],target_seq_data[train_size:]

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return (train_loader,test_loader,source_mapping,target_mapping)
    @classmethod
    def train(cls,model:TransformerModel,train_loader:DataLoader,test_loader:DataLoader,source_mapping,target_mapping,epochs:int=100,lr:float=0.001,file_name:str='model.pth'):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(),lr=lr)
        best_loss = float('inf')
        for epoch in range(epochs):
            # 训练模型
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                # 使用前一时间步的输入作为目标序列
                src = batch_x.permute(1, 0, 2)
                tgt = batch_y.permute(1, 0, 2)
                tgt = torch.zeros(batch_y.shape).permute(1,0,2)
                output = model(src,tgt)
                train_loss = criterion(output, batch_y.squeeze(-1))
                train_loss.backward()
                optimizer.step()            
            # 验证模型
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    src = batch_x.permute(1, 0, 2)
                    tgt = batch_y.permute(1, 0, 2)
                    tgt = torch.zeros(batch_y.shape).permute(1,0,2)
                    output = model(src,tgt)
                    loss = criterion(output, batch_y.squeeze(-1))
                    total_loss += loss.item()
            if (epoch + 1) % 1 == 0: 
                print(f'Epoch [{epoch + 1}/{epochs}],Train Loss: {train_loss.item():.4f} , Val Loss: {total_loss / len(test_loader):.4f}')
            if total_loss < best_loss:
                best_loss = total_loss
                best_model = model
                torch.save({"mapping":{"source":source_mapping,"target":target_mapping},
                            "paramers":best_model.paramers,
                            "model":best_model.state_dict()},file_name)  
    @classmethod
    def load_model(cls,file_name:str="model.pth",mapping=None):
        model_dict = torch.load(file_name)
        print(model_dict.keys())
        print("mapping=",model_dict["mapping"])
        print("paramers=",model_dict["paramers"])
        paramers = model_dict["paramers"]
        model = TransformerModel(**paramers,mapping=mapping) 
        model.load_state_dict(model_dict["model"])
        return model
    @classmethod
    def predict(cls,model:TransformerModel,loader:DataLoader):
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in loader:
                src = batch_x.permute(1,0,2)
                tgt = torch.zeros(batch_y.shape).permute(1,0,2)
                output = model(src, tgt)
                if model.mapping:
                    mean = model.mapping[1]['zd']['mean'] 
                    std = model.mapping[1]['zd']['std'] 
                    print("\nbatch_y=",(batch_y*std)+mean,"\noutput=",(output*std)+mean)
                else:
                    print("\nbatch_y=",batch_y,"\noutput=",output)
        return output
    
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # 创建一个 (max_len, d_model) 的矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        pe = pe.unsqueeze(0)  # 增加一个维度，使其形状为 (1, max_len, d_model)

        self.register_buffer('pe', pe)  # 注册为 buffer，不会被视为模型参数

    def forward(self, x):
        # x 的形状为 (seq_len, batch_size, d_model)
        x = x + self.pe[:, :x.size(1), :]  # 将位置编码加到输入上
        return self.dropout(x)

if __name__ == '__main__':
    import sys
    import sqlite3
    args = sys.argv
    if len(args)<2:
        print("usage: python torch_lib.py train|predict")
        exit(1)
    conn = sqlite3.connect("daily_pro.db")
    df = pd.read_sql("select * from daily where code='000001'",conn)
    #df=pd.DataFrame({"c":range(300),"zd":range(300)})
    epochs = 1000
    lr = 0.001
    batch_size = 200
    source_seq = 10
    target_seq = 1
    d_model=64
    key_column='date'
    source_column=['o','h','l','c','v','v_status','c_status','hs','zf','ma5c','ma10c','macd','szc','szv','ma5szc','ma10szc','szmacd']
    source_dim=len(source_column)
    target_column=['zd']
    target_dim=len(target_column)
    
    if args[1].lower()=='train':
        train_loader,test_loader,source_mapping,target_mapping = TorchLib.get_dataloader(
                df,source_column,target_column,key_column,source_seq=source_seq,target_seq=target_seq,batch_size=batch_size,filter=[])
        print("source_mapping=",source_mapping)
        print("target_mapping=",target_mapping)
        model = TransformerModel(source_dim=source_dim,target_dim=target_dim,d_model=d_model)
        TorchLib.train(model,train_loader,test_loader,source_mapping,target_mapping,epochs,lr)
    else:
        filter = ['2024-12-30','2024-12-31','2025-01-02']
        train_loader,test_loader,source_mapping,target_mapping = TorchLib.get_dataloader(
                df,source_column,target_column,key_column,source_seq=source_seq,target_seq=target_seq,batch_size=1,split=1,filter=filter)
        model = TorchLib.load_model("model.pth",mapping=(source_mapping,target_mapping)) 
        pred = TorchLib.predict(model,train_loader)
    
    
        