from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from PIL import Image
import io
import model
import logging


logger = logging.getLogger(__name__)

app = FastAPI()



logging.basicConfig(filename='API.log', level=logging.INFO)
logger.info('Started')

security = HTTPBasic()

VALID_USERS = {
    "admin": "password",
    "user1": "secret123",
    "user2": "letmein"
}


cls_model = model.load_pytorch_model()
logger.info("model_fetched")


def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    
    
    
    if credentials.username not in VALID_USERS or VALID_USERS[credentials.username] != credentials.password:
        logging.info(f"INVALID USER CREDS : {credentials.username}:{credentials.password}")    
        
        
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    logging.info(f"VALID USER CREDS : {credentials.username}")
    return credentials.username

@app.post("/predict/")
async def predict(file: UploadFile = File(...),username: str = Depends(authenticate) ):
    
    try:
        contents = await file.read()
        logging.info(f"file read : ")
    except Exception as e:
        logging.info(f"Error reading file: {e}")
        
    
    image = Image.open(io.BytesIO(contents)).convert("L")
    
    # Predict using model
    try:
        logging.info("Predicting Class...")
        predicted_class = model.predict(image=image, model=cls_model)
        logging.info(f"Predicted class: {predicted_class} ")
    
    except Exception as e  :
        logging.info(f"Error predicting class: {e}")
        predicted_class = "NotAvailable"
        
        
    return {"username": username, "prediction": predicted_class}

# uvicorn app:app --host 0.0.0.0 --port 8080
