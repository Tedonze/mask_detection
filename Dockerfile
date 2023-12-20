FROM python:3.11-slim

COPY .  /mask_detection 

WORKDIR /mask_detection

RUN pip install poetry && \ 
    poetry install && \ 
    poetry shell

EXPOSE 8501

ENTRYPOINT [ "streamlit" , "run", "src/main.py" ]