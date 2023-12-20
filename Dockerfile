FROM python:3.11-slim

COPY .  /mask_detection 

WORKDIR /mask_detection

RUN pip install poetry && \ 
    poetry install 
    

EXPOSE 8501

ENTRYPOINT [ "poetry" ,"run","streamlit" , "run", "src/main.py" ]