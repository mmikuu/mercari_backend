import os, sys
import sqlite3
import codecs
import pandas as pd
from collections import deque
from tap import Tap


class Args(Tap):
    database: str = "sql/wishlists.db"
    output_path: str = ""
    search_word: str = ""
    
class Exp:
    def __init__(self, args: Args):
        self.args: Args = args
        
        self.database: str = args.database
        self.search_word = args.search_word
        self.output_path = args.output_path



    def run(self):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()

        new = codecs.open(self.output_path, mode='w', encoding='UTF-8')
        new.write(self.columns)

        search = cur.execute('SELECT CUI, AUI, SAB, CODE, STR FROM ENGJPN WHERE CUI == ?', [(cui)])
        result = search.fetchall()

        new.close()
        cur.close()
        conn.close()

def main(args: Args):
    print('PROBLEM CHECK:', args)
    exp = Exp(args=args)
    exp.run()

if __name__ == "__main__":
    args = Args().parse_args()
    main(args)