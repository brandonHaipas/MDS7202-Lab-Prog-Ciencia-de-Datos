import pickle
import pandas as pd
from fastapi import FastAPI

# crear aplicación
app = FastAPI()

@app.get("/")
def home():
    return "Esta api dispone un modelo de clasificación supervisado para predecir si el agua es potable a partir de mediciones químicas recolectadas por sensores IoT instalados en la red de distribución de Maipú. El modelo recibe como entrada nueve variables (como pH, dureza, sólidos disueltos, entre otras) y entrega como salida un valor binario que indica si el agua es apta (1) o no apta (0) para el consumo humano, permitiendo alertar oportunamente ante riesgos sanitarios."

@app.post("/potabilidad")
async def potabilidad(ph: float, Hardness: float, Solids: float, Chloramines: float, Sulfate: float, Conductivity: float, Organic_carbon: float, Trihalomethanes: float, Turbidity: float):
    with open('models/exp_461236034205691343/model.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict(
        pd.DataFrame(
            [{
                "ph": ph,
                "Hardness": Hardness,
                "Solids": Solids,
                "Chloramines": Chloramines,
                "Sulfate": Sulfate,
                "Conductivity": Conductivity,
                "Organic_carbon": Organic_carbon,
                "Trihalomethanes": Trihalomethanes,
                "Turbidity": Turbidity
            }]
        )
    )

    return {"Potability": int(prediction[0])}

if __name__ == '__main__':
    uvicorn.run('fastapi_app:app', port = 8000)