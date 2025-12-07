# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import time
import os
import joblib

# Define the path where the trained model artifact is expected
MODEL_PATH = os.path.join("models", "rec_model.pkl")

# --- Model Loading and Placeholder Logic ---

def load_recommendation_model():
    """
    Loads the trained model from the 'models' directory.
    If the model doesn't exist (e.g., in a development environment),
    it returns a simple placeholder function.
    """
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading trained model from {MODEL_PATH}...")
            # In a real scenario, you would use the loaded model (e.g., model.predict)
            model = joblib.load(MODEL_PATH)
            # For this example, we still return the simplified function
            print("Model loaded successfully.")
            return lambda user_id: [f"item_{user_id}_A_model_v{model.version}", f"item_{user_id}_B_model_v{model.version}"]
        except Exception as e:
            print(f"Error loading model: {e}. Using placeholder.")
            pass # Fall through to placeholder
            
    print("WARNING: Model not found or failed to load. Using placeholder recommendations.")
    # Placeholder function for simplicity or if the model failed to load
    return lambda user_id: [f"item_{user_id}_A", f"item_{user_id}_B", f"item_{user_id}_C"]


RECOMMENDATION_MODEL = load_recommendation_model()

# --- FastAPI Initialization ---

app = FastAPI(
    title="Real-time Recommendation API",
    description="Serves personalized item recommendations.",
    version="1.0.0"
)

# Pydantic model for request validation
class UserQuery(BaseModel):
    user_id: int
    context_data: dict = {}

# --- Endpoints ---

@app.get("/health")
def health_check():
    """Health check endpoint for ops/monitoring."""
    return {"status": "ok", "model_version": "v1.0"}

@app.post("/recommend")
def get_recommendations(query: UserQuery):
    """
    Generates real-time recommendations for a given user ID.
    """
    start_time = time.time()
    
    # Use the loaded model/placeholder to generate recommendations
    recommendations = RECOMMENDATION_MODEL(query.user_id)
    
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "user_id": query.user_id,
        "recommendations": recommendations,
        "latency_ms": round(latency_ms, 2)
    }

if __name__ == "__main__":
    import uvicorn
    # Command to run the app: python app.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
