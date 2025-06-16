from utils import table_parameters, cprint
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("nickypro/tinyllama-110M")
tokenizer = AutoTokenizer.from_pretrained("nickypro/tinyllama-110M")

prompt = "The princess and the castle"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
cprint(generated_text)
