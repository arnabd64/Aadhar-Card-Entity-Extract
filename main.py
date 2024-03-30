import uvicorn
import os
from src.api.handlers import download_assets

if __name__ == "__main__":
    download_assets()
    uvicorn.run("src.api.server:server",
                host = "0.0.0.0",
                port = int(os.getenv("PORT", 8000)),
                workers=int(os.getenv("WORKERS", 1)))
