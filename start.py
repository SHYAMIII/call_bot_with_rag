
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8002))
    
    print(f"🚀 Starting AI Call Manager on http://{host}:{port}")
  
    # Start the server
    uvicorn.run(
        "main:app",
        host=host, 
        port=port,
        reload=True,
        log_level="info"
    ) 