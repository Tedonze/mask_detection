import os
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()


class GlobalSettings(BaseModel):
    s3_endpointUrl: str = os.getenv('S3_ENDPOINT_URL')
    s3_key: str = os.getenv('S3_KEY')
    s3_secret: str = os.getenv('S3_SECRET')
    s3_token: str = os.getenv('S3_TOKEN')
    s3_prefix: str = 'ikonkobo/covid19_pis_data/'


settings = GlobalSettings()
