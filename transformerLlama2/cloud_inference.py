import modal
import os 
from pathlib import Path 
import torch 

app = modal.App("llama-scratch")

image = modal.Image.debian_slim().pip_install(
    "torch",
    "sentencepiece",
    "huggingface_hub[hf_transfer]",
    "transformers",
    "numpy"
    ).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    ).add_local_file(local_path="./core.py",remote_path="/root/core.py"
    ).add_local_file(local_path="./block_utils.py",remote_path="/root/block_utils.py"
    ).add_local_file(local_path="../pos_freqs.py",remote_path="/root/pos_freqs.py"
    ).add_local_dir(local_path="../mha", remote_path="/root/mha")
    

# create a Volume, or retrieve it if it exists
volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
MODEL_DIR = Path("/models")

class CONFIG:
    VOCAB: int = 32_000
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


@app.function(
    volumes={MODEL_DIR: volume},  # "mount" the Volume, sharing it with your function
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
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
    import subprocess
    subprocess.run(["ls", "-lh", "/models/meta-llama/Llama-2-7b"])
    print(f"Model downloaded to {MODEL_DIR / repo_id}")

@app.cls(
        gpu="H100", 
        volumes={MODEL_DIR: volume}, 
        image=image
        )
class PIPELINE:
    repo_id: str="meta-llama/Llama-2-7b"
    WEIGHTS = MODEL_DIR / repo_id / "consolidated.00.pth"
    TOK = MODEL_DIR / repo_id / "tokenizer.model"
    device="cuda"

    @modal.enter()
    def enter(self):
        from core import TransformerLlama2
        from block_utils import load_weights_into_llama
        import subprocess 
        subprocess.run("nvidia-smi")
        weights = torch.load(self.WEIGHTS, weights_only=True)
        model = TransformerLlama2(CONFIG,device="cuda")
        load_weights_into_llama(model, CONFIG, weights)
        self.model = model.to(self.device)

        import sentencepiece as spm 
        sp = spm.SentencePieceProcessor()
        sp.load(str(self.TOK))
        self.tokenizer = sp

    @modal.method()
    def inference(self):
        prompt = "the evolution of the british empire across the world"
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

@app.local_entrypoint()
def main():
    download_model.remote()
    pipeline = PIPELINE()
    pipeline.inference.remote()

