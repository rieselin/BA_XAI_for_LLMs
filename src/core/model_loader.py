import threading

_model      = None
_tokenizer  = None
_model_lock = threading.Lock()

def get_model(instruct: bool = True):
    """Load model once, thread-safely, and return it."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    with _model_lock:
        if _model is not None:       # Double-Checked Locking
            return _model, _tokenizer
        model_name = (
            "unsloth/Llama-3.1-8B-Instruct-bnb-4bit" if instruct
            else "unsloth/Llama-3.1-8B-bnb-4bit"
        )
        from unsloth import FastLanguageModel
        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
            max_seq_length=8192,
            device_map="cuda",
        )
        FastLanguageModel.for_inference(_model)
    return _model, _tokenizer