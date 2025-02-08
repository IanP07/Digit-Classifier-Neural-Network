import torch
import io
from fastapi import FastAPI, UploadFile
from PIL import Image
from torchvision import transforms
from torch import nn, load
from fastapi.middleware.cors import CORSMiddleware



# Define the FastAPI app
app = FastAPI()

# Allows 405 methods
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to restrict access if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)


# Load trained model
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, x):
        return self.model(x)

# Load model
model = ImageClassifier()
model.load_state_dict(load("model_state.pt", map_location=torch.device('cpu')))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))    # normalizes values between 0 and 1
])

@app.post("/predict/")
async def predict(file: UploadFile):
    try:
        # Read image
        image = Image.open(io.BytesIO(await file.read()))
        image = transform(image).unsqueeze(0)  # Makes it a 784 item list

        # Make prediction
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output).item()

        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}

# Keeps render backend alive
@app.post("/keep-alive/")
async def keep_alive():
    return {"status": "alive"}


# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
