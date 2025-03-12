import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import os
import model
import requests
import io


# Set up Streamlit page config
st.set_page_config(page_title="Digit Drawing App", layout="centered")

st.title("Draw a Digit and get it recognized")

# --- Sidebar for credentials ---

        
        

# --- Canvas for drawing ---
canvas_result = st_canvas(
    fill_color="black",  # Background color
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# --- Save drawing ---
if st.button("Save Drawing"):
    if canvas_result.image_data is not None:
        # Convert to PIL Image and save
        img = Image.fromarray((canvas_result.image_data[:, :, :3]).astype('uint8'))
        
        img_path = "./input_digits/digit.jpg"
        st.session_state['img_path'] = img_path
        if os.path.exists(img_path): 
            os.remove(img_path)
        img.save(img_path)
        
        st.session_state['image'] = img
        
        st.success("Image saved as 'digit.jpg'")
        st.image(img, caption="Saved Digit")

            
temp_username = st.text_input("Username", key="temp_username")
temp_password = st.text_input("Password", type="password", key="temp_password")


# --- Predict button ---
if st.button("Predict"):
    
    
    
    
    if 'img_path' in st.session_state:
        img_path = st.session_state['img_path']
        if img_path is not None:

            # ✅ Save credentials into session state
            
            if temp_username and temp_password:
                st.session_state['username'] = temp_username
                st.session_state['password'] = temp_password
                st.success("Credentials saved!")
            else:
                st.error("Please enter both username and password.")

            
            
            
            
            
            # 🔒 Ensure user authentication before predicting
            if ('username' in st.session_state) and ('password' in st.session_state):
                
                
                
                
                
                
                with open(img_path, "rb") as img_file:
                    # Send request to FastAPI with Basic Auth
                    response = requests.post(
                        "http://localhost:8080/predict/",
                        files={"file": img_file},
                        auth=(st.session_state['username'], st.session_state['password'])
                    )


                if response.status_code == 200:
                    predicted_class = response.json().get("prediction")
                    st.success("Model predicted successfully!")
                    st.write(f"Predicted class: **{predicted_class}**")

                    img = st.session_state['image']
                    st.image(img, caption="Saved Digit")
                elif response.status_code == 401:
                    st.error("❌ Invalid credentials.")
                else:
                    st.error(f"❌ Error: {response.status_code} - {response.text}")
            else:
                st.error("❌ Please provide valid credentials before predicting.")