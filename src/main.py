import streamlit as st
from PIL import Image
from model.net_model import Net
from model.base_model import BaseModel
from pandas import DataFrame


net = Net(2)
model = BaseModel(net)
model.load_model(path='checkpoint/loss_validation0.042.save')
st.title('Mask detector ')

st.title('by Tedonze and Konkobo :sunglasses:')

img_file_buffer = st.camera_input('Upload a PNG image',)
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    st.write(DataFrame(model.predict(image), columns=['no_mask', 'with mask']))










