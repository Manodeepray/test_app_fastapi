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

#database containing userid and passwords

VALID_USERS = {
    "admin": "password",
    "user1": "secret123",
    "user2": "letmein"
}


cls_model = model.load_pytorch_model() #load Resnet18 Model

logger.info("model_fetched")


def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Authenticate a user using HTTP Basic Authentication.

    Args:
        credentials (HTTPBasicCredentials): The username and password provided in the request.

    Returns:
        str: The username if authentication is successful.

    Raises:
        HTTPException: If the provided credentials are invalid (status code 401).

    Steps:
        1. Check if the username exists in `VALID_USERS` and if the password matches.
        2. Log invalid credentials if authentication fails.
        3. Raise an HTTP 401 error if credentials are invalid.
        4. Log valid credentials and return the username if authentication succeeds.
    """
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



async def predict(file: UploadFile = File(...), username: str = Depends(authenticate)):
    """
    Predict the class of an uploaded image using a pre-trained model.

    Args:
        file (UploadFile): The image file uploaded by the user.
        username (str): The authenticated username (obtained via dependency injection).

    Returns:
        dict: A dictionary containing the username and the predicted class.

    Steps:
        1. Read the contents of the uploaded file.
        2. Convert the image to grayscale and preprocess it.
        3. Use the model to predict the class of the image.
        4. Handle errors during file reading or prediction gracefully.
    """
    try:
        contents = await file.read()  # Read the uploaded file
        logging.info("File read successfully.")
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        raise HTTPException(status_code=400, detail="Error reading uploaded file.")

    image = Image.open(io.BytesIO(contents)).convert("L")  # Convert image to grayscale

    # Predict using the model
    try:
        logging.info("Predicting class...")
        predicted_class = model.predict(image=image, model=cls_model)  # Perform prediction
        logging.info(f"Predicted class: {predicted_class}")
    except Exception as e:
        logging.error(f"Error predicting class: {e}")
        predicted_class = "NotAvailable"  # Default value in case of prediction failure

    return {"username": username, "prediction": predicted_class}

# uvicorn app:app --host 0.0.0.0 --port 8080
