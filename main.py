from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

# Import routers and dependencies
# from api import router as auth_router
# from database.middleware import verify_token
from app import get_tax_assistant




load_dotenv()

app = FastAPI(
    title="Python Learning Assistant",
    description="Ask python related questions",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# app.include_router(auth_router)



class TaxQuestion(BaseModel):
    question: str

class TaxImpactRequest(BaseModel):
    monthly_income: float





@app.get("/")
def home():
    return {
        "message": "Welcome to Python Dojo",
        
    }



@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Python guru is up and running"
    }





@app.post("/ask")
def ask_tax_question(
    request: TaxQuestion,
    # user_info: dict = Depends(verify_token)
):
    """Ask a coding related question"""
    try:
        
        assistant = get_tax_assistant()
        # user_id = f"user_{user_info['user_id']}"
        user_id = f"user_1"

        
        response = assistant.ask_question(
            question=request.question,
            user_id=user_id
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "question": request.question,
            "answer": response
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )