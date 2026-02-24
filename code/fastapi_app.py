from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
import boto3
import json

app = FastAPI()

ENDPOINT_NAME = "sales-prediction-endpoint"
REGION = "us-east-2"

HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Sales Prediction</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 500px; margin: 50px auto; padding: 20px; }}
        h1 {{ color: #333; }}
        label {{ display: block; margin-top: 15px; font-weight: bold; }}
        input, select {{ width: 100%; padding: 8px; margin-top: 5px; box-sizing: border-box; }}
        button {{ margin-top: 20px; padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; font-size: 16px; width: 100%; }}
        button:hover {{ background: #0056b3; }}
        .result {{ margin-top: 20px; padding: 15px; background: #e8f5e9; border-radius: 5px; font-size: 20px; color: #2e7d32; text-align: center; }}
        .error {{ margin-top: 20px; padding: 15px; background: #ffebee; border-radius: 5px; color: #c62828; }}
    </style>
</head>
<body>
    <h1>Sales Prediction</h1>
    <form action="/predict" method="post">
        <label>Store ID</label>
        <input type="number" name="store_id" value="1" required />

        <label>Product Category</label>
        <select name="product_category">
            <option>Electronics</option>
            <option>Clothing</option>
            <option>Food</option>
            <option>Sports</option>
            <option>Home</option>
        </select>

        <label>Units Sold</label>
        <input type="number" name="units_sold" value="5" required />

        <label>Unit Price ($)</label>
        <input type="number" step="0.01" name="unit_price" value="299.99" required />

        <label>Discount (0.0 to 1.0)</label>
        <input type="number" step="0.01" min="0" max="1" name="discount_pct" value="0.1" required />

        <label>Is Weekend?</label>
        <select name="is_weekend">
            <option value="0">No</option>
            <option value="1">Yes</option>
        </select>

        <button type="submit">Predict Sales</button>
    </form>
    {result_html}
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_FORM.format(result_html="")

@app.get("/predict", response_class=HTMLResponse)
def predict_get():
    return RedirectResponse(url="/")

@app.post("/predict", response_class=HTMLResponse)
def predict(
    store_id: int = Form(...),
    product_category: str = Form(...),
    units_sold: int = Form(...),
    unit_price: float = Form(...),
    discount_pct: float = Form(...),
    is_weekend: int = Form(...)
):
    try:
        payload = {
            "store_id": store_id,
            "product_category": product_category,
            "units_sold": units_sold,
            "unit_price": unit_price,
            "discount_pct": discount_pct,
            "is_weekend": is_weekend
        }

        client = boto3.client("sagemaker-runtime", region_name=REGION)
        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload)
        )

        result = json.loads(response["Body"].read().decode())
        prediction = result["prediction"][0]
        result_html = f'<div class="result">Predicted Sales: <strong>${prediction:,.2f}</strong></div>'

    except Exception as e:
        result_html = f'<div class="error">Error: {str(e)}</div>'

    return HTML_FORM.format(result_html=result_html)
