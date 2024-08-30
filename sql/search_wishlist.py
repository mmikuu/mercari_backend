import os, sys
import sqlite3


def get_matched_data(your_list: dict) -> int:
    database: str = "wishlists.db"
    conn = sqlite3.connect(database)
    cur = conn.cursor()

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
    return min, max, people


def add_listings(your_list: dict) -> None:
    database: str = "wishlists.db"
    conn = sqlite3.connect(database)
    cur = conn.cursor()

    search = cur.execute('SELECT * FROM wishlist WHERE items_name == ?', ([your_list["items_name"]])) # 型番が一致するものをwishlistから検索
    result = search.fetchall()

    sql = ''' INSERT INTO listings(id,category,items_name,storage)
             VALUES(?,?,?,?) ''' # 出品者の情報をlistingsにINSERT
    cur = conn.cursor()
    cur.execute(sql, your_list)
    conn.commit()

    search = cur.execute('SELECT * FROM listings')
    result = search.fetchall()
    print(result)

    cur.close()
    conn.close()