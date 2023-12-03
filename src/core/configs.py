import os
from pydantic import BaseModel
from dotenv import load_dotenv
from s3fs import S3FileSystem


load_dotenv()


class GlobalSettings(BaseModel):
    s3_endpointUrl: str = os.getenv('S3_ENDPOINT_URL')
    s3_key: str = os.getenv('S3_KEY')
    s3_secret: str = os.getenv('S3_SECRET')
    s3_token: str = os.getenv('S3_TOKEN')
    s3_prefix: str = os.getenv('S3_PREFIX')


settings = GlobalSettings()


fs = S3FileSystem(
    client_kwargs={'endpoint_url': settings.s3_endpointUrl},
    key=settings.s3_key,
    secret=settings.s3_secret,
    token=settings.s3_token
)
