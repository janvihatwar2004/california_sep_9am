
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import joblib

app = FastAPI()

# Connect frontend files
templates = Jinja2Templates(directory="templates")
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

# Load the trained model
obj = joblib.load("california_data2.joblib")
model = obj["Model"]
columns = obj["Columns"]

print("Model columns:", columns)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "columns": columns,
            "prediction": None,
            "error": None,
            "values": {}
        }
    )


@app.post("/", response_class=HTMLResponse)
async def predict(request: Request):
    form_data = await request.form()
    inputs = []
    values = {}
    prediction = None
    error = None

    try:
        for column in columns:
            raw_value = str(form_data.get(column, "")).strip()
            values[column] = raw_value

            if not raw_value:
                raise ValueError(f"Please enter a value for {column}.")

            value = float(raw_value)

            if not np.isfinite(value):
                raise ValueError(f"{column} must be a valid number.")

            inputs.append(value)

        result = model.predict(np.array(inputs).reshape(1, -1))
        prediction = float(result[0])

    except ValueError as exc:
        error = str(exc)

    except Exception:
        app.logger.exception("Prediction failed")
        error = "Prediction failed. Please check your inputs."

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "columns": columns,
            "prediction": prediction,
            "error": error,
            "values": values
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
