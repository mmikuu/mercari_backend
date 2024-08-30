import os, sys
import sqlite3
import codecs
import pandas as pd
from collections import deque
from tap import Tap


class Args(Tap):
    database: str = "sql/wishlists.db"
    wishlist: dict = {"id":0, "category":0, "item_name":0, "min_budget":0, "max_budget":0}
    wishlist_path: str = "sql/wishlist.json"
    ### to do
    ### 受け取りは、dict型？jsonのpath？確定しておくこと！
    
class Exp:
    def __init__(self, args: Args):
        self.args: Args = args        
        self.database: str = args.database
        self.wishlist: dict = args.wishlist
        self.wishlist_path: str = args.wishlist_path

    def run(self):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()

        wishlist_item = [v[1] for i, v in enumerate(self.wishlist.items())]

        sql = ''' INSERT INTO wishlist(id,category,items_name,min_budget,max_budget)
                 VALUES(?,?,?,?,?) '''
        cur = conn.cursor()
        cur.execute(sql, wishlist_item)
        conn.commit()

        search = cur.execute('SELECT * FROM wishlist')
        result = search.fetchall()
        print(result)

        cur.close()
        conn.close()

def main(args: Args):
    print('PROBLEM CHECK:', args)
    exp = Exp(args=args)
    exp.run()

if __name__ == "__main__":
    args = Args().parse_args()
    main(args)