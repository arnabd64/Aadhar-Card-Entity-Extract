import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "src.server:server",
        host = "0.0.0.0",
        port = 8000,
        workers = 4,
        loop = "uvloop",
        http = "httptools" 
    )