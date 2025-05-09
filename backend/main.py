# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
import sqlite3
import os
import uuid
from datetime import datetime

# Create the FastAPI app
app = FastAPI(title="Suggestions API", description="API for storing form submissions")

# Add CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
# Create the database and table if they don't exist
def init_db():
    if not os.path.exists("suggestions.db"):
        conn = sqlite3.connect("suggestions.db")
        cursor = conn.cursor()
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
        conn.commit()
        conn.close()

# Initialize database on startup
@app.on_event("startup")
async def startup():
    init_db()

# Define the request model for validation
class SuggestionCreate(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    phone_number: str = Field(..., min_length=10, max_length=20)
    suggestion: str = Field(..., min_length=10, max_length=1000)

# Define the response model
class SuggestionResponse(BaseModel):
    id: str
    first_name: str
    last_name: str
    email: str
    phone_number: str
    suggestion: str
    created_at: str

# POST endpoint to create a new suggestion
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

# GET endpoint to retrieve all suggestions (for admin purposes)
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

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)