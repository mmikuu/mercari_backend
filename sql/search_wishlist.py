import os, sys
import sqlite3
import codecs
import pandas as pd
from collections import deque
from tap import Tap


class Args(Tap):
    database: str = "sql/wishlists.db"
    your_list: dict = {"id":0, "category":0, "items_name":0, "storage":0} #出品者の情報

    
class Exp:
    def __init__(self, args: Args):
        self.args: Args = args        
        self.database: str = args.database
        self.your_list: dict = args.your_list

    def run(self):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()

        search = cur.execute('SELECT * FROM wishlist WHERE items_name == ?', ([self.your_list["items_name"]])) # 型番が一致するものをwishlistから検索
        result = search.fetchall()

        matched = [] ## matchしたものを格納する
        for item in result:
            storage = item[6] #wishlistのストレージ
            if storage == self.your_list["storage"]:
                matched.append(item) ## list(tuples)
        
        people = len(matched) #欲しがってる人の人数を設定
        max = 0
        min = 10000000 # 欲しがってる人の金額範囲を設定
        for item in matched:
            if max < item[9]:
                max = item[9]
            if min > item[9]:
                min = item[9]

        sql = ''' INSERT INTO listings(id,category,items_name,storage)
                 VALUES(?,?,?,?) ''' # 出品者の情報をlistingsにINSERT
        cur = conn.cursor()
        cur.execute(sql, self.your_list)
        conn.commit()

        search = cur.execute('SELECT * FROM listings')
        result = search.fetchall()
        print(result)

        cur.close()
        conn.close()
        return min,max,people #予算の最低額、予算の最高額、欲しがってる人数

def main(args: Args):
    print('PROBLEM CHECK:', args)
    exp = Exp(args=args)
    exp.run()

if __name__ == "__main__":
    args = Args().parse_args()
    main(args)