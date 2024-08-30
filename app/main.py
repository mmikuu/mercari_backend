from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from googleapiclient.discovery import build
from dotenv import load_dotenv
import numpy as np
import requests



import sql ##榎原の関数
import sqlite3

#matched_wish_yours = None
#min = None
#max = None



wish_list = [
    {
    "title": "素敵な旅行",
    "min_budget": 50000,
    "max_budget": 200000,
    "description": "国内か海外でリラックスできる、思い出に残る旅。温泉や自然を楽しめる場所が理想です。"
    },
    {
    "title": "最新のスマートフォン",
    "min_budget": 30000,
    "max_budget": 120000,
    "description": "カメラ性能が良く、バッテリー持ちが良い。デザインもシンプルで使いやすいモデル。"
    }
]

# FastAPIのインスタンス作成
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Pre-trained model (ResNet50 for image encoding)
image_model = models.resnet50(pretrained=True)
image_model.eval()

# Sentence Transformer for text encoding
text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Transformation for input image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/contents")
async def add_listinglist(
    device: Optional[str] = Form(None),
    type: Optional[str] = Form(None),
    storage: Optional[str] = Form(None),
    ):
    form_data = {
        "id":0,
        "category":device,
        "items_name":type,
        "storage":storage,
    }

    print(form_data)

    #input:出品者の管理情報　-> output:wishlistとの一致情報
    matched_wish_yours = sql.get_matched_data(form_data) #榎原が書いたやつ
    # 入力は{"id":0, "category":0, "items_name":0, "storage":0}の形式にして入れてね！
    # (min, max, cnt on people)で出力されます。
    return matched_wish_yours



@app.get("/budget")
async def get_budget():
    return {"matched_wish_yours":matched_wish_yours[0][2],"min":matched_wish_yours[0],"max":matched_wish_yours[1]}


@app.post("/wishes")
async def reccomend_wishlist(
    category: Optional[str] = Form(None),
    wanna: Optional[str] = Form(None),
    max_budget: Optional[str] = Form(None),
    min_budget: Optional[str] = Form(None),
    storage: Optional[str] = Form(None),
    speed: Optional[str] = Form(None),
    camera: Optional[str] = Form(None),
    size: Optional[str] = Form(None),
    ):  # フォームデータを辞書にまとめる
    form_data = {
        "id":0,
        "category" : category,
        "items_name": wanna,
        "storage": storage,
        "min-budget": min_budget,
        "max-budget": max_budget,
    }

    print(form_data)
    # wishlistにINSERT
    insert_data = sql.add_wishlist(form_data) #### databaseにINSERT

    # 画像の読み込みと前処理
    image = Image.open(img_file.file)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

        


    # 画像から特徴ベクトルを取得
    with torch.no_grad():
        features = image_model.avgpool(image_model.layer4(image_model.layer3(image_model.layer2(image_model.layer1(image_model.maxpool(image_model.relu(image_model.bn1(image_model.conv1(input_batch))))))))).squeeze()
    future_vector = features.numpy()

    # 必要に応じて次元を調整（ここでは先に得られるベクトルをそのまま利用）
    if future_vector.shape[0] > 384:
        future_vector = future_vector[:384]
    elif future_vector.shape[0] < 384:
        future_vector = np.pad(future_vector, (0, 384 - future_vector.shape[0]))

    # wish_listのテキストをベクトル化
    wish_vectors = [text_model.encode(f"{wish['title']} {wish['description']}") for wish in wish_list]

    # 類似度の計算
    similarities = cosine_similarity([future_vector], wish_vectors)[0]

    # 最も類似度の高いwishを取得
    most_similar_index = np.argmax(similarities)
    most_similar_wish = wish_list[most_similar_index]

    
    return JSONResponse(content={
        "most_similar_wish": most_similar_wish,
        "similarity_score": float(similarities[most_similar_index])
    })




@app.get("/health")
async def healthcheck():
    return {"message": "ok"}

@app.get("/search")
async def search(query: str):
    query = query + "説明書 pdf"
    results = google_search(query, API_KEY, cse_id, num=5)
    pdf = None
    for result in results:
        if result['link'].split('.')[-1] == 'pdf':
            url = result['link']
            get_pdf(url)
            break
    
    return JSONResponse(content={"results": results})



def google_search(query, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, **kwargs).execute()
    return res['items']

def get_pdf(url):
        
    response = requests.get(url)

    # ステータスコードを確認
    if response.status_code == 200:
        # ファイルをバイナリモードで保存
        with open("downloaded_sample.pdf", "wb") as file:
            file.write(response.content)
        print("PDFファイルが保存されました。")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="debug", reload=True)
