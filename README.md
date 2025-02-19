# Random-Forrest-Gump
Machine Learning in Production - Random Forrest Gump's (Team 22) repo for the Movie Recommendations Project

## How to Run

### Setting Up the Environment
Create a virtual environment and install the required packages:
```bash
uv venv -p 3.10
source .venv/bin/activate
pip install -r requirements.txt
```

### Training the Model
Run the training script:
```bash
python train.py
```

### Running the API
In one terminal:
```bash
python model.py
```

In another terminal:
```bash
uvicorn main:app --host 0.0.0.0 --port 8082
```
or if you want to see the prediction results:
```bash
uvicorn main:app --host 0.0.0.0 --port 8082 --log-level debug
```