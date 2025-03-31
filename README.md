# About
This repository is a from scratch implementation of LLaMA2-3 models.
It is still under development but when it is done, it will include inference and training
code together.
For now, model_inference.py and inference.py are implemented which can be used to create a model for inference and
model can be sampled with top p or greedy strategies.
You can see the implementations of RoPE(Rotary Positional Encodings), KV cache,
GQA(Grouped Query Attention), SwiGLU activation which are essential components of modern LLMs.
For now, I will implement single gpu training and inference. Later I might modify it for
multi gpu training.