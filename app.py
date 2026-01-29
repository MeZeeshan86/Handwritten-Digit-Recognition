import streamlit as st
import numpy as np
import torch
from PIL import Image,ImageOps
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from model import CNN

st.set_page_config(page_title="Hand-Written Digit Recognizer",layout="centered")

model=CNN()
model.load_state_dict(torch.load("model/mnist_cnn.pth", map_location="cpu"))
model.eval()

st.markdown("""
<h1 style="text-align:center;">Handwritten Digit Recognizer</h1>
<p style="text-align:center;color:gray;">Draw a digit (0–9)</p>
""",unsafe_allow_html=True)

left,right=st.columns([1,1])

with left:
    canvas=st_canvas(
        fill_color="white",
        stroke_width=12,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas"
    )

with right:
    pred_box=st.empty()
    chart_box=st.empty()

def preprocess_pil(img):
    img=img.convert("L")
    img=ImageOps.invert(img)
    arr=np.array(img)
    coords=np.column_stack(np.where(arr>0))

    if coords.size:
        y_min,x_min=coords.min(axis=0)
        y_max,x_max=coords.max(axis=0)
        arr=arr[y_min:y_max+1,x_min:x_max+1]
    else:
        return None

    img=Image.fromarray(arr)
    img.thumbnail((20,20),Image.Resampling.LANCZOS)

    canvas28=Image.new("L",(28,28),0)
    canvas28.paste(img,((28-img.width)//2,(28-img.height)//2))

    arr=np.array(canvas28).astype("float32")/255.0
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)

# Live prediction
if canvas.image_data is not None:
    raw=canvas.image_data.astype("uint8")
    if np.sum(raw[:,:,0]<250)>20:
        pil_img=Image.fromarray(raw)
        x=preprocess_pil(pil_img)

        if x is not None:
            with torch.no_grad():
                out=model(x)
                probs=torch.softmax(out,dim=1)[0].numpy()
                pred=probs.argmax()

            pred_box.markdown(
                f"<h2 style='text-align:center;'>Prediction: <span style='color:#4CAF50'>{pred}</span></h2>",
                unsafe_allow_html=True
            )

            df=pd.DataFrame({
                "Probability":probs,
                "Digit":[str(i) for i in range(10)]
            }).set_index("Digit")

            chart_box.bar_chart(df,horizontal=True)
        else:
            pred_box.info("Draw a digit")
            chart_box.empty()
    else:
        pred_box.info("Draw a digit")
        chart_box.empty()
else:
    pred_box.info("Draw a digit")
    chart_box.empty()

if __name__ == "__main__":
    import os
    import streamlit.web.bootstrap as bootstrap

    port = int(os.environ.get("PORT", 8501))

    bootstrap.run(
        "app.py",
        "",
        [],
        flag_options={
            "server.port": port,
            "server.address": "0.0.0.0",
            "server.headless": True,
        },
    )
