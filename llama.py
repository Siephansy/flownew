from transformers import AutoTokenizer, AutoModelForCausalLM

# Escolha o modelo desejado
model_name = "meta-llama/Llama-2-7b-chat-hf"

# Baixe o modelo e o tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True  # Ativa quantização
)

print("Modelo LLaMA baixado com sucesso!")
