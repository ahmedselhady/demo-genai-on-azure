import torch

FILES_DIR:str = "../../data/"
CHUNK_SIZES_CHARS:int = 500
CHUNKS_OVERLAP_CHARS:int = 50
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"

DB_NAME = "local_db"

GENERATION_TEMP = 0.01
MAX_GENERATION_NEW_TOKENS = 256

MODEL_TYPE = "LLAMA" # MISTRAL or LLAMA
MODELS_PATH = "../../model/"
DOWNLOAD_SAMPLE = True

DEEPL_AUTH_KEY = "1e6ca761-1246-4ab0-8814-1eaefe9a3575:fx"  # Replace with your key
HF_TOEKN = "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"