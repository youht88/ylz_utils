import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import os

def main():
    device = "cuda:0" if torch.cuda.is_available() else \
             "mps" if torch.mps.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else \
                  torch.float16 if torch.mps.is_available() else torch.float32

    #model_id = "openai/whisper-large-v3"
    model_id = "openai/whisper-small"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    sample = dataset[0]["audio"]
    
    generate_kwargs = {
        "max_new_tokens": 256,
        "num_beams": 1,
        "condition_on_prev_tokens": False,
        "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "return_timestamps": True
    }

    #result = pipe(sample, return_timestamps=True)
    while True:
        text = input("请输入文本：")
        if text == "/q":
            break
        elif os.path.exists(f"{os.path.dirname(os.path.abspath(__file__))}/{text}.m4a"):
            print(f"解析{text}.m4a")
            result = pipe(f"{os.path.dirname(os.path.abspath(__file__))}/{text}.m4a",return_timestamps=True)
            print(result["chunks"])
        else:
            break

if __name__ == "__main__":
    main()
