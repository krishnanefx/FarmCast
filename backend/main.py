import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from geopy.geocoders import Nominatim

app = FastAPI()

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

geolocator = Nominatim(user_agent="pangu-weather-app")

class WeatherRequest(BaseModel):
    location: str
    model: str

@app.get("/")
async def read_index():
    index_path = os.path.join(frontend_dir, "index.html")
    if not os.path.exists(index_path):
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("Frontend not built yet.", status_code=404)
    return FileResponse(index_path)

def get_grid_idx(lat, lon):
    lat_idx = int(round((90.0 - lat) / 0.25))
    lat_idx = max(0, min(lat_idx, 720))
    
    lon_360 = lon if lon >= 0 else lon + 360
    lon_idx = int(round(lon_360 / 0.25))
    lon_idx = max(0, min(lon_idx, 1439))
    
    return lat_idx, lon_idx

def load_model(model_hours: str):
    valid_models = {"1", "3", "6", "24"}
    if model_hours not in valid_models:
        raise ValueError(f"Invalid model hours: {model_hours}")
    
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"pangu_weather_{model_hours}.onnx")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return session

@app.post("/api/weather")
async def get_weather(req: WeatherRequest):
    try:
        location = geolocator.geocode(req.location)
        if not location:
            raise HTTPException(status_code=404, detail="Location not found")
        
        lat, lon = location.latitude, location.longitude
        lat_idx, lon_idx = get_grid_idx(lat, lon)
        
        try:
            session = load_model(req.model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
            
        dummy_input = np.zeros((5, 13, 721, 1440), dtype=np.float32)
        dummy_input_surface = np.zeros((4, 721, 1440), dtype=np.float32)
        
        outputs = session.run(None, {
            'input': dummy_input,
            'input_surface': dummy_input_surface
        })
        
        output_surface = outputs[1]
        
        mslp = float(output_surface[0, 0, lat_idx, lon_idx])
        u10 = float(output_surface[0, 1, lat_idx, lon_idx])
        v10 = float(output_surface[0, 2, lat_idx, lon_idx])
        t2m = float(output_surface[0, 3, lat_idx, lon_idx])
        
        return {
            "location": location.address,
            "coordinates": {"lat": lat, "lon": lon},
            "grid_idx": {"lat_idx": lat_idx, "lon_idx": lon_idx},
            "metrics": {
                "temperature_2m": t2m,
                "u_wind_10m": u10,
                "v_wind_10m": v10,
                "mean_sea_level_pressure": mslp
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
