import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from fastapi.responses import HTMLResponse
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
import joblib
laptop_data = pd.read_csv('lop.csv')
encoder_brand = LabelEncoder()
encoder_processor = LabelEncoder()
encoder_os = LabelEncoder()
laptop_data_encoded = laptop_data.copy()
laptop_data_encoded['brand'] = encoder_brand.fit_transform(laptop_data_encoded['brand'].astype(str))
laptop_data_encoded['processor_brand'] = encoder_processor.fit_transform(laptop_data_encoded['processor_brand'].astype(str))
laptop_data_encoded['os'] = encoder_os.fit_transform(laptop_data_encoded['os'].astype(str))
laptop_data_encoded['ram_gb'] = laptop_data_encoded['ram_gb'].str.replace(' GB', '').astype(int)
laptop_data_encoded['graphic_card_gb'] = laptop_data_encoded['graphic_card_gb'].replace(' GB', '', regex=True).astype(float)
features = ['brand', 'ram_gb', 'processor_brand', 'graphic_card_gb', 'os']
X = laptop_data_encoded[features]
y = laptop_data_encoded['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
joblib.dump(knn_model, 'laptop_price_model.pkl')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryModel(BaseModel):
    brand: str = None
    ram_gb: int = None
    processor_brand: str = None
    graphic_card_gb: float = None
    os: str = None

@app.post("/query_laptops/")
async def query_laptops(query: QueryModel):
    default_brand = 'dell'
    default_processor_brand = 'intel'
    default_os = 'windows'

    brand_no = {'asus': 1, 'avita': 2, 'dell': 3, 'hp': 4, 'lenovo': 5, 'msi': 6, 'acer': 7}
    proc_no = {'intel': 1, 'amd': 0}
    os_no = {'': 0, 'mac': 1, 'windows': 2,}
    brand = query.brand.lower() if query.brand else default_brand
    ram_gb = query.ram_gb 
    processor_brand = query.processor_brand.lower() if query.processor_brand else default_processor_brand
    graphic_card_gb = query.graphic_card_gb 
    os = query.os.lower() if query.os else default_os
    if ram_gb < 0 or graphic_card_gb < 0:
        raise HTTPException(status_code=400, detail="RAM and graphic card GB must be non-negative.")

    try:
        brnd = brand_no[brand]
        pro = proc_no[processor_brand]
        os_encoded = os_no[os]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid value for {str(e)}")

    input_data = pd.DataFrame({
        'brand': [brnd],
        'ram_gb': [ram_gb],
        'processor_brand': [pro],
        'graphic_card_gb': [graphic_card_gb],
        'os': [os_encoded]
    })

    prediction = knn_model.predict(input_data)
    predicted_price = prediction[0]
    
    return HTMLResponse(content=f"predicted_price : {predicted_price:.2f}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
