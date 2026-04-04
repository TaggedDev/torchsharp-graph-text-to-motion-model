"""
One-time script to export CLIP ViT-B/32 text encoder to ONNX format.

Prerequisites:
    pip install torch
    pip install git+https://github.com/openai/CLIP.git

Output:
    models/clip/clip_text_encoder.onnx    (~170 MB)
    models/clip/bpe_simple_vocab_16e6.txt.gz  (~1.3 MB)
"""

import os
import shutil

import clip
import torch


class ClipTextEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.model = clip_model

    def forward(self, input_ids):
        return self.model.encode_text(input_ids)


def main():
    model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
    model.eval()

    wrapper = ClipTextEncoder(model)
    dummy = torch.zeros(1, 77, dtype=torch.long)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "models", "clip")
    os.makedirs(out_dir, exist_ok=True)

    onnx_path = os.path.join(out_dir, "clip_text_encoder.onnx")
    print(f"Exporting CLIP text encoder to {onnx_path} ...")

    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["input_ids"],
        output_names=["text_embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "text_embeddings": {0: "batch"},
        },
        opset_version=14,
    )
    print(f"ONNX model saved ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")

    # Copy BPE vocabulary file for the C# tokenizer
    clip_pkg_dir = os.path.dirname(clip.__file__)
    vocab_src = os.path.join(clip_pkg_dir, "bpe_simple_vocab_16e6.txt.gz")
    vocab_dst = os.path.join(out_dir, "bpe_simple_vocab_16e6.txt.gz")
    shutil.copy(vocab_src, vocab_dst)
    print(f"Vocabulary copied to {vocab_dst}")

    # Validation: encode a test sentence and save for cross-checking with C#
    import numpy as np

    text = clip.tokenize(["a man walks forward"])
    with torch.no_grad():
        embedding = model.encode_text(text).cpu().numpy()
    ref_path = os.path.join(out_dir, "reference_embedding.npy")
    np.save(ref_path, embedding)
    print(f"Reference embedding saved to {ref_path} (shape: {embedding.shape})")
    print("Done.")


if __name__ == "__main__":
    main()
