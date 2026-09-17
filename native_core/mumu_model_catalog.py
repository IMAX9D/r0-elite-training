"""Explicit compatible checkpoint choices; no arbitrary pickle loading or fallback."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'models/hokoff-restart319051/runtime'
DEFAULT_MODEL = 'restart319051'
MODELS = {
    'restart319051': ('最新 · 重启后 319,051 步', 'hokoff-restart319051/inference.pt',
                      'hokoff-restart319051/encoder-contract.json'),
    'step1037042': ('上一版 · 1,037,042 步', 'hokoff-releases/hokoff-bc-step1037042.pt',
                    'hokoff-releases/encoder-contract.json'),
    'step785157': ('更早版 · 785,157 步', 'hokoff-releases/hokoff-bc-step785157.pt',
                   'hokoff-releases/encoder-contract.json'),
}

def model_paths(model_id):
    _, checkpoint, contract = MODELS[model_id]
    return ROOT / 'models' / checkpoint, ROOT / 'models' / contract, SOURCE

def available_models():
    return {key: value[0] for key, value in MODELS.items()
            if all(path.exists() for path in model_paths(key))}

def identify_checkpoint(path):
    if path:
        target = Path(path).resolve()
        for key in MODELS:
            if model_paths(key)[0].resolve() == target:
                return key
    return None
