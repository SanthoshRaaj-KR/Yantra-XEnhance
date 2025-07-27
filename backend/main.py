import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# --- 1. Initialize FastAPI App ---
app = FastAPI(
    title="X-Ray Denoising API",
    description="An API to classify noise and denoise X-ray images.",
    version="1.0.0",
)

# --- 2. Set up CORS ---
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Load AI Models ---
def load_all_models():
    """Loads the classifier and all denoising models."""
    print("Loading AI models...")
    try:
        classifier_model = tf.keras.models.load_model('models/xray_noise_classifier_resnet50v2.keras')
        denoiser_models = {
            'gaussian': tf.keras.models.load_model('models/gaussian_denoiser_final_model.keras'),
            'poisson': tf.keras.models.load_model('models/poisson_denoising.keras'),
            'salt_pepper': tf.keras.models.load_model('models/salt_pepper_denoiser.keras'),
            'speckle': tf.keras.models.load_model('models/speckle_denoising_final_model.keras')
        }
        print("✅ Models loaded successfully!")
        return classifier_model, denoiser_models
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None

CLASSIFIER, DENOISERS = load_all_models()
NOISE_CLASSES = ['gaussian', 'poisson', 'salt_pepper', 'speckle']

# --- 4. Define Helper Functions ---
def preprocess_image(image_bytes: bytes):
    """Converts image bytes to a NumPy array for the models."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # --- THIS IS THE CRUCIAL FIX ---
        # 1. Convert to RGB for 3 color channels.
        img = img.convert('RGB')
        # 2. Resize to 224x224, the exact size the ResNet50 model expects.
        img = img.resize((224, 224))
        
        img_array = np.array(img)
        # Add the batch dimension. The channel dimension is now 3.
        img_array = img_array[np.newaxis, ...] 
        return img_array / 255.0
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file. Could not preprocess. Error: {e}")

def postprocess_output(denoised_array: np.ndarray):
    """Converts model output array back to an image file in memory."""
    # Squeeze the array to remove the batch dimension
    processed_array = np.squeeze(denoised_array)
    # Denormalize from 0-1 to 0-255 and convert to integer type
    processed_array = (processed_array * 255).astype(np.uint8)
    image = Image.fromarray(processed_array)
    
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    return img_io

# --- 5. Create the API Endpoint ---
@app.post("/api/denoise", response_class=StreamingResponse)
async def denoise_image(image: UploadFile = File(...)):
    """
    Receives an X-ray image, classifies the noise, applies the correct
    denoiser model, and returns the cleaned image.
    """
    if not CLASSIFIER or not DENOISERS:
        raise HTTPException(status_code=503, detail="Models are not available on the server.")

    image_bytes = await image.read()

    try:
        # Preprocess for the classifier
        classifier_input = preprocess_image(image_bytes)
        
        # Run the classifier
        prediction = CLASSIFIER.predict(classifier_input)
        noise_type_index = np.argmax(prediction)
        noise_type = NOISE_CLASSES[noise_type_index]
        print(f"Detected noise type: {noise_type}")

        # --- IMPORTANT ---
        # We need to re-process the image for the denoiser models if they
        # expect a different input size or format (e.g., grayscale 256x256).
        # Assuming denoisers expect grayscale 256x256 for this example.
        img_for_denoiser = Image.open(io.BytesIO(image_bytes)).convert('L').resize((256, 256))
        denoiser_input = np.array(img_for_denoiser)[np.newaxis, ..., np.newaxis] / 255.0

        # Select and run the correct denoiser
        denoiser_model = DENOISERS[noise_type]
        denoised_array = denoiser_model.predict(denoiser_input)

        output_image_buffer = postprocess_output(denoised_array)

        return StreamingResponse(output_image_buffer, media_type="image/png")

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

# --- 6. Add a root endpoint for basic health check ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the X-Ray Denoising API!"}
