import uvicorn

from forvis.web.app import app


def test_web():
    # Start the FastAPI application
    uvicorn.run("forvis.web.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    test_web()
