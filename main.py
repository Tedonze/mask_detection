from pydantic import BaseModel, Field
from fastapi import FastAPI, File, UploadFile
import uuid

app = FastAPI()


class Image(BaseModel):
    url: str = Field(pattern= r'/^https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)$/')
    proba: float = Field(gt=0, lt=1)


@app.get('/')
def root():
    return {
        'hello': 'Welcome to Ted&KO Classifier'
    }


@app.get("/upload/")
async def create_upload_file(file: UploadFile = File(...)):

    file.filename = f"{uuid.uuid64()}.jpg"
    content = await file.read()

    with open(f"/{file.filename}", 'wb') as f:
        f.write(content)
    return file.filename







