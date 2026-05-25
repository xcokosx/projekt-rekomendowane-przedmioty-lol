from app.neural_network import ItemRanker
from app.config import MODEL_PATH

def test_model_loads():
    model = ItemRanker.load(MODEL_PATH)
    assert model is not None
    assert hasattr(model, "forward")
