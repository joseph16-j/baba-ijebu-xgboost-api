# Premier Lotto Data Lab — Baba Ijebu XGBoost API

Flask + XGBoost inference API for the Wix prediction console.

## Deploy on Render

1. Create a **private** GitHub repository and upload these three files:
   - `ai_lotto_production_system.py`
   - `requirements.txt`
   - `render.yaml`
2. In Render, choose **New → Blueprint**, connect that repository, and apply the `render.yaml` configuration.
3. Render builds the Python service and exposes an HTTPS URL. Check `/api/health` first.
4. Enter the resulting base URL in the XGBoost console's API connection field in Wix.

## API

- `GET /api/health`
- `POST /api/v1/ai/predict`

POST JSON:

```json
{
  "profile": "National",
  "depth": 100,
  "historical_records": [
    {"game": "National", "date": "2026-10-30", "w": [1, 2, 3, 4, 5], "m": [6, 7, 8, 9, 10]}
  ]
}
```

The example row above only illustrates the request schema. It is not production training data. The service trains only on valid records supplied by the Wix Saved Results snapshot or a configured history JSON file. At least three valid draws are required for the selected game profile.

## Important deployment notes

- Render's free web service sleeps after 15 minutes without inbound traffic and takes about a minute to wake. Its filesystem is ephemeral, so model pickle files are a cache only and are recreated as needed.
- Render describes free instances as suitable for testing and hobby projects, not production. Upgrade the service before relying on it for uninterrupted user-facing availability.
- Do not put API secrets in browser code. For production, route calls through a Wix backend function and keep any shared API credential in Wix Secrets Manager and Render environment variables.
- Lottery results are random. The scores are experimental rankings, not guaranteed outcomes.
