# Chay HUFLIT RAG Robot qua LM Studio

1. Mo LM Studio, tai va load mot chat model.
2. Trong tab **Developer**, bat Local Server o cong `1234`.
3. Cai thu vien va chay API:

```powershell
pip install -r requirements.txt
python api_server.py
```

Ung dung mac dinh goi `http://localhost:1234/v1`. Co the cau hinh bang bien
moi truong truoc khi chay:

```powershell
$env:LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
$env:LM_STUDIO_MODEL = "ten-model-trong-LM-Studio"
$env:LM_STUDIO_API_KEY = "lm-studio"
$env:LM_STUDIO_TIMEOUT = "120"
python api_server.py
```

`LM_STUDIO_MODEL` la tuy chon. Neu bo trong, ung dung se chon model dau tien
do endpoint `/v1/models` cua LM Studio tra ve.
