import os
import logging
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
import tensorflow as tf
import numpy as np
import cv2
import sqlite3
from datetime import datetime

app = FastAPI(title="X-Ray Image Denoiser & Suggestions", 
              description="API for X-Ray Image Denoising and Form Submissions")

# Add CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Ensure required directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load pre-trained models
try:
    # Load Keras ResNet50V2 classification model instead of PyTorch model
    noise_classifier = tf.keras.models.load_model('model/xray_noise_classifier_resnet50v2.keras')
    
    # Load Keras denoising models
    gaussian_denoiser = tf.keras.models.load_model('model/gaussian_denoiser_final_model.keras')
    poisson_denoiser = tf.keras.models.load_model('model/poisson_denoising.keras')
    salt_pepper_denoiser = tf.keras.models.load_model('model/salt_pepper_denoiser.keras')
    speckle_denoiser = tf.keras.models.load_model('model/speckle_denoising_final_model.keras')

    print("All models loaded successfully!")

except Exception as e:
    print(f"Error loading models: {e}")
    noise_classifier = None
    gaussian_denoiser = None
    poisson_denoiser = None
    salt_pepper_denoiser = None
    speckle_denoiser = None

# Serve static files for output images
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Database Initialization
def init_db():
    """Initialize SQLite database for suggestions"""
    conn = sqlite3.connect("suggestions.db")
    cursor = conn.cursor()
    
    # Suggestions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS suggestions (
        id TEXT PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT NOT NULL,
        phone_number TEXT NOT NULL,
        suggestion TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    ''')
    
    # Image processing log table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS image_processing_log (
        id TEXT PRIMARY KEY,
        original_filename TEXT NOT NULL,
        processed_filename TEXT NOT NULL,
        noise_type TEXT NOT NULL,
        processed_at TEXT NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
@app.on_event("startup")
async def startup():
    init_db()

# Suggestion-related Pydantic Models
class SuggestionCreate(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    phone_number: str = Field(..., min_length=10, max_length=20)
    suggestion: str = Field(..., min_length=10, max_length=1000)

class SuggestionResponse(BaseModel):
    id: str
    first_name: str
    last_name: str
    email: str
    phone_number: str
    suggestion: str
    created_at: str

# Image Processing Helper Functions
def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess the image for model input
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # <-- read directly as grayscale
    img = cv2.resize(img, (256, 256))  # if needed
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)   # shape: (256, 256, 1)
    img = np.expand_dims(img, axis=0)  
    return img

def preprocess_for_classification(image_path, target_size=(224, 224)):
    """
    Preprocess image for the ResNet50V2 classification model
    ResNet50V2 typically expects 224x224 RGB images
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to read image file")
    
    # Ensure we have a 3-channel image for ResNet50V2
    if len(img.shape) == 2:  # If grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Resize to expected input size for ResNet50V2 (typically 224x224)
    img = cv2.resize(img, target_size)
    
    # Convert to float32 and normalize
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def classify_noise_type(image_path):
    """
    Classify the type of noise in the image using Keras ResNet50V2 model
    """
    if noise_classifier is None:
        raise HTTPException(status_code=500, detail="Noise classifier model not loaded")
    
    # Preprocess the image for Keras ResNet50V2
    img = preprocess_for_classification(image_path)
    
    # Perform inference
    predictions = noise_classifier.predict(img)
    
    # Get the predicted class
    noise_classes = ['gaussian', 'poisson', 'salt_pepper', 'speckle']
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = noise_classes[predicted_class_index]
    
    return predicted_class

def denoise_image(image_path, noise_type):
    """
    Denoise the image using the appropriate model based on noise type
    """
    # Select the appropriate denoiser based on noise type
    denoiser_map = {
        'gaussian': gaussian_denoiser,
        'poisson': poisson_denoiser,
        'salt_pepper': salt_pepper_denoiser,
        'speckle': speckle_denoiser
    }
    
    denoiser = denoiser_map.get(noise_type)
    
    if denoiser is None:
        raise HTTPException(status_code=500, detail=f"No denoiser found for {noise_type} noise")
    
    # Preprocess the image for the denoising model
    preprocessed_img = preprocess_image(image_path)
    
    # Denoise the image
    denoised_img = denoiser.predict(preprocessed_img)
    
    # Post-process the denoised image
    denoised_img = np.squeeze(denoised_img)  # Remove batch and channel dimensions
    denoised_img = (denoised_img * 255).astype(np.uint8)  # Scale back to [0, 255]
    
    return denoised_img

# Suggestion Endpoints
@app.post("/api/suggestions/", response_model=SuggestionResponse)
async def create_suggestion(suggestion: SuggestionCreate):
    try:
        # Connect to the database
        conn = sqlite3.connect("suggestions.db")
        cursor = conn.cursor()
        
        # Generate a unique ID and timestamp
        suggestion_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        # Insert the suggestion into the database
        cursor.execute(
            '''
            INSERT INTO suggestions (id, first_name, last_name, email, phone_number, suggestion, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                suggestion_id, 
                suggestion.first_name, 
                suggestion.last_name, 
                suggestion.email, 
                suggestion.phone_number, 
                suggestion.suggestion,
                created_at
            )
        )
        
        # Commit the transaction
        conn.commit()
        
        # Create the response
        response = SuggestionResponse(
            id=suggestion_id,
            first_name=suggestion.first_name,
            last_name=suggestion.last_name,
            email=suggestion.email,
            phone_number=suggestion.phone_number,
            suggestion=suggestion.suggestion,
            created_at=created_at
        )
        
        # Close the connection
        conn.close()
        
        return response
    
    except Exception as e:
        # Handle any errors
        raise HTTPException(status_code=500, detail=f"Error creating suggestion: {str(e)}")

@app.get("/api/suggestions/", response_model=list[SuggestionResponse])
async def get_suggestions():
    try:
        # Connect to the database
        conn = sqlite3.connect("suggestions.db")
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()
        
        # Get all suggestions
        cursor.execute("SELECT * FROM suggestions ORDER BY created_at DESC")
        rows = cursor.fetchall()
        
        # Convert to response model
        suggestions = [
            SuggestionResponse(
                id=row["id"],
                first_name=row["first_name"],
                last_name=row["last_name"],
                email=row["email"],
                phone_number=row["phone_number"],
                suggestion=row["suggestion"],
                created_at=row["created_at"]
            )
            for row in rows
        ]
        
        # Close the connection
        conn.close()
        
        return suggestions
    
    except Exception as e:
        # Handle any errors
        raise HTTPException(status_code=500, detail=f"Error retrieving suggestions: {str(e)}")

# Image Denoising Endpoint
@app.post("/api/denoise")
async def denoise_image_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to handle image denoising:
    1. Save uploaded image
    2. Classify noise type 
    3. Apply appropriate denoising model
    4. Save and return processed image
    """
    try:
        # Check if models are loaded
        if noise_classifier is None or gaussian_denoiser is None:
            raise HTTPException(status_code=500, detail="Models not properly loaded")
            
        # Generate unique filename
        file_extension = file.filename.split('.')[-1]
        filename = f"{uuid.uuid4()}.{file_extension}"
        input_path = os.path.join("uploads", filename)
        
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Classify noise type
        noise_type = classify_noise_type(input_path)
        print(f"Detected noise type: {noise_type}")
        
        # Denoise image
        denoised_img = denoise_image(input_path, noise_type)
        
        # Save denoised image to outputs folder
        output_filename = f"denoised_{filename}"
        output_path = os.path.join("outputs", output_filename)
        cv2.imwrite(output_path, denoised_img)
        
        # Log the image processing in the database
        conn = sqlite3.connect("suggestions.db")
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO image_processing_log (id, original_filename, processed_filename, noise_type, processed_at)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (
                str(uuid.uuid4()), 
                filename, 
                output_filename, 
                noise_type, 
                datetime.now().isoformat()
            )
        )
        conn.commit()
        conn.close()
        
        # Return the path to the processed image (to be displayed in frontend)
        return {
            "processed_url": f"/outputs/{output_filename}", 
            "noise_type": noise_type
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Simple health check endpoint to verify the API is running
    """
    return {"status": "alive", "models_loaded": noise_classifier is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)