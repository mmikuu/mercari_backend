from fastapi import FastAPI, File, UploadFile
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
import openai


# .envファイルを読み込み
load_dotenv()

# 環境変数からAPIキーとCSE IDを取得
API_KEY = os.getenv("API_KEY")
cse_id = os.getenv("CSE_ID")
openai.api_key = "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" 

cse_id = "YOUR_CSE_ID"
API_KEY = "YOUR_API_KEY"


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
async def reccomend_wishlist(
    img_file: UploadFile = File(...),
    ):
    # 画像の読み込みと前処理
    image = Image.open(img_file.file)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    database_data = from_database("sql/wishlists.db")#### database入れ込み

    get_answer_item_detail()


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



def get_answer_item_detail


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
