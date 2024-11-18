import sqlite3
import pandas as pd
import akshare as ak

if __name__=="__main__":
    daily_db=sqlite3.connect("daily.db")
    info_df=ak.stock_info_a_code_name()
    codes=info_df.code.to_list()
    conn = sqlite3.connect("daily.db")
    from tqdm import tqdm
    with tqdm(total=len(codes)) as pbar:
        for code in codes:
            df=ak.stock_zh_a_hist(code,start_date='20220101')
            df.to_sql("daily",if_exists="append",index=True,con=conn)
            pbar.update(1)
