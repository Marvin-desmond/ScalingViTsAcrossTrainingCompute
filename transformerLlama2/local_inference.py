from pathlib import Path 
import torch 
import sys
import os 
from dotenv import load_dotenv 
load_dotenv()

MODEL_DIR = Path("./models")

class CONFIG:
    VOCAB: int = 2_000
    CONTEXT_LEN: int = 4096 
    DIM: int = 4096  
    N_HEADS: int = 32
    N_LAYERS: int = 32
    HIDDEN_DIM: int = 11008
    DTYPE: torch.dtype = torch.bfloat16 

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)
        if eos_id is not None and (idx_next == eos_id).all():
            break
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)
    return idx

def download_model(
    repo_id: str="meta-llama/Llama-2-7b",
    revision: str=None,  # include a revision to prevent surprises!
    ):
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=repo_id, 
        local_dir=MODEL_DIR / repo_id,
        token=os.environ["HF_TOKEN"]
        )
    print(f"Model weights and tokenizer downloaded to {MODEL_DIR / repo_id}")

class PIPELINE:
    repo_id: str="meta-llama/Llama-2-7b"
    WEIGHTS = MODEL_DIR / repo_id / "consolidated.00.pth"
    TOK = MODEL_DIR / repo_id / "tokenizer.model"
    device="cpu"

    def __init__(self,device):
        self.device = device
        from core import TransformerLlama2
        from block_utils import load_weights_into_llama
        weights = torch.load(self.WEIGHTS, weights_only=True)
        model = TransformerLlama2(CONFIG,device=self.device)
        print(model)
        import sys 
        sys.exit(0)
        load_weights_into_llama(model, CONFIG, weights)
        self.model = model.to(self.device)
        import sentencepiece as spm 
        sp = spm.SentencePieceProcessor()
        sp.load(str(self.TOK))
        self.tokenizer = sp

    def inference(self):
        prompt = "The story of the boy who turned to a robot"
        encoded = self.tokenizer.encode_as_ids(prompt)
        encoded = torch.tensor(encoded).unsqueeze(0)
        tokens = encoded.to(self.device)
        tokens = generate(
                model=self.model,
                idx=tokens,
                max_new_tokens=500,
                context_size=CONFIG.CONTEXT_LEN,
                top_k=40,
                temperature=0.7,
            )
        decoded_text = self.tokenizer.decode_pieces(tokens.squeeze(0).tolist())
        print(decoded_text)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else (
               "mps" if torch.backends.mps.is_available() else "cpu"))
    download_model()
    if not torch.cuda.is_available():
        sys.exit(0)
    pipeline = PIPELINE(device=device)
    pipeline.inference()

