# streamlit_app/utils/api_client.py
import requests

def predict_price(backend_base_url: str, payload: dict, timeout=10):
    url = backend_base_url.rstrip("/") + "/predict"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
