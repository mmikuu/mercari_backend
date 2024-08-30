import os, sys
import sqlite3

def add_wishlist(wishlist: dict) -> None: #### wishlistに追加する関数
    database: str = "wishlists.db"
    conn = sqlite3.connect(database)
    cur = conn.cursor()
    wishlist_item = [v[1] for i,v in enumerate(wishlist.items())]

    sql = ''' INSERT INTO wishlist(id,category,items_name,storage,min_budget,max_budget)
             VALUES(?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, wishlist_item)
    conn.commit()

    search = cur.execute('SELECT * FROM wishlist')
    result = search.fetchall()
    print(result)

    cur.close()
    conn.close()

def add_listings(your_list: dict) -> None:  ##listingsに追加する関数
    database: str = "wishlists.db"
    conn = sqlite3.connect(database)
    cur = conn.cursor()

    search = cur.execute('SELECT * FROM wishlist WHERE items_name == ?', ([your_list["items_name"]])) # 型番が一致するものをwishlistから検索
    result = search.fetchall()

    your_item = [v[1] for i,v in enumerate(your_list.items())]
    sql = ''' INSERT INTO listings(id,category,items_name,storage)
             VALUES(?,?,?,?) ''' # 出品者の情報をlistingsにINSERT
    
    cur.execute(sql, your_item)
    conn.commit()

    search = cur.execute('SELECT * FROM listings')
    result = search.fetchall()
    print(result)

    cur.close()
    conn.close()


def get_matched_data(your_list: dict) -> tuple:  ### 出品者がwishlistとのマッチを確認する関数
    database: str = "wishlists.db"
    conn = sqlite3.connect(database)
    cur = conn.cursor()

    x = cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print('database table:', x.fetchall())
    
    search = cur.execute('SELECT * FROM wishlist WHERE items_name == ?', ([your_list["items_name"]])) # 型番が一致するものをwishlistから検索
    result = search.fetchall()

    matched = [] ## matchしたものを格納する
    for item in result:
        storage = item[6] #wishlistのストレージ
        if storage == your_list["storage"]:
            matched.append(item) ## list(tuples)
        
    people = len(matched) #欲しがってる人の人数を設定
    max = 0
    min = 10000000 # 欲しがってる人の金額範囲を設定
    for item in matched:
        if max < item[9]:
            max = item[9]
        if min > item[9]:
            min = item[9]
    print(matched)
    return min, max, people