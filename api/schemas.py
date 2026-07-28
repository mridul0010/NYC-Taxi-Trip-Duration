from pydantic import BaseModel, Field

class TaxiPredictionRequest(BaseModel):
    vendor_id: int = Field(..., description="Vendor ID (e.g., 1 or 2)")
    pickup_datetime: str = Field(..., description="ISO timestamp format (YYYY-MM-DD HH:MM:SS)")
    dropoff_datetime: str = Field(..., description="ISO timestamp format (YYYY-MM-DD HH:MM:SS)")
    passenger_count: int = Field(..., description="Number of passengers")
    pickup_longitude: float = Field(..., description="Pickup Longitude")
    pickup_latitude: float = Field(..., description="Pickup Latitude")
    dropoff_longitude: float = Field(..., description="Dropoff Longitude")
    dropoff_latitude: float = Field(..., description="Dropoff Latitude")
    store_and_fwd_flag: str = Field(..., description="Y or N flag")
