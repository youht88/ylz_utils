import warnings
import numpy as np
import pandas as pd

from ylz_utils.torch.transformer.transformer import TransformerModel


if __name__ == '__main__':
    import sys
    import sqlite3
    args = sys.argv
    if len(args)<2:
        print("usage: python torch_lib.py train|predict|test")
        exit(1)


    conn = sqlite3.connect("daily_pro.db")
    df = pd.read_sql("select * from daily where code='000001'",conn)
    #df=pd.DataFrame({"c":range(300),"zd":range(300)})
    key_column=None #'date'
    source_column=['o','h','l','c','v','hs','zf','ma5c','ma10c','szc','szv','ma5szc','ma10szc']
    target_column=['zd']
    source_seq = 10
    target_seq = 1
    
    
    df=pd.DataFrame({"a":np.random.rand(300)*1000,"b":np.random.rand(300)*1000,"c":np.random.rand(300)*1000})
    df['diff'] = df['a'] - df['b']
    df["z"] = df['diff'].shift(1).rolling(window=10).std()
    key_column=None #'date'
    source_column=['a','b']
    target_column=['z']
    source_seq = 10 
    target_seq = 1
   

    d_model=64
    nhead = 4
    n_layers = 6
    norm_type = 'std'
    batch_size = 800
    epochs = 50
    lr = 0.001
    warnings.filterwarnings("ignore")

    print(df.head(50))
    if args[1].lower()=='train':
        try:
            model = TransformerModel(source_column=source_column,target_column=target_column,key_column=key_column,
                                    source_seq = source_seq,target_seq = target_seq,
                                    nhead=nhead,n_layers=n_layers,
                                    d_model=d_model,norm_type=norm_type)
            model.set_dataloader(df,batch_size=batch_size,filter=[])
            model.model_train(epochs = epochs,lr = lr)
        except Exception as e:
            print(f"ERROR is {e}")
            print(f"YOU should define a pd.DataFrame with columns and seq which defined by below arguments:")
            print(f"key_column is {model.key_column}")
            print(f"source_column is {model.source_column} and source_seq is {model.source_seq}")
            print(f"source_column is {model.target_column} and source_seq is {model.target_seq}")
    elif args[1].lower()=='predict':
        try:
            #filter = ['2024-12-30','2024-12-31','2025-01-02','2025-01-08']
            filter = [20,50,100,150,200,250]
            model:TransformerModel = TransformerModel.load_model("model.pth") 
            model.set_dataloader(df,batch_size=1,split=1,filter=filter)
            pred = model.model_predict()
        except Exception as e:
            print(f"ERROR is {e}")
            print(f"YOU should define a pd.DataFrame with columns and seq which defined by below arguments:")
            print(f"key_column is {model.key_column}")
            print(f"source_column is {model.source_column} and source_seq is {model.source_seq}")
            print(f"source_column is {model.target_column} and source_seq is {model.target_seq}")
    else:
        try:
            df = pd.DataFrame({"input":["美国总统是谁?","厦门今天天气如何?"],"output":["特朗普","晴朗"]})
            key_column=None #'date'
            source_column=['input']
            target_column=['output']
            source_seq = 10
            target_seq = 1
            d_model = 512
            
            model = TransformerModel(source_column=source_column,target_column=target_column,key_column=key_column,
                                    source_seq = source_seq,target_seq = target_seq,
                                    d_model=d_model)
            model.set_dataloader(df,batch_size=batch_size,filter=[])
            model.model_train(epochs = epochs,lr = lr)

            # input = torch.LongTensor([[0,2,0,5],[4,3,2,9]])
            # d_model = 512
            # vocab = 1000
            # embedding = Embedding(d_model,vocab)
            # emb = embedding(input)
            # print(emb,"\n",emb.shape)
        except Exception as e:
            print(f"ERROR is {e}")
            
