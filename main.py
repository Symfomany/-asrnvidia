from fastapi import FastAPI

app = FastAPI()


def read_root():
    return {"Hello": "World!"}
