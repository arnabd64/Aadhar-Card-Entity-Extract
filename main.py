import uvicorn
import os

if __name__ == "__main__":
    uvicorn.run("src.server:server",
                host = "0.0.0.0",
                port = int(os.getenv("PORT", 8000)),
                workers = int(os.getenv("WORKERS", 1)),
                loop = "uvloop",
                http = "httptools")
