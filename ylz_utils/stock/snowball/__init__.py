from fastapi import FastAPI

from ylz_utils.stock.snowball.snowball import SnowballStock

class SnowballLib():    
    def __init__(self):
        self.snowball = SnowballStock()

    def register_app(self,app:FastAPI): 
        app.include_router(self.snowball.router,prefix="/snowball")

