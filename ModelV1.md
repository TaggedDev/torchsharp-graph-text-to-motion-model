This model is a text-conditioned DDPM pipeline: the input prompt is tokenized and encoded into a 512-dim CLIP embedding, reverse diffusion starts from Gaussian noise in motion feature space `[1, T, 263]`, and at each timestep the `GraphDenoiser` predicts noise using timestep conditioning plus a 4-layer graph convolution stack over the 22-joint skeleton before producing the final motion sequence `[1, T, 263]`.

```mermaid
flowchart TD
    A["Text Prompt<br/>Type: string<br/>In: raw text<br/>Out: raw text"] --> B["CLIP Tokenizer<br/>In: string<br/>Out: [1, 77] token ids"]
    B --> C["CLIP Text Encoder<br/>In: [1, 77]<br/>Out: [1, 512]"]

    N["Gaussian Noise x_T<br/>Type: initial motion latent<br/>In: sampled noise<br/>Out: [1, T, 263]"] --> S
    C --> S
    T0["Diffusion Timestep t<br/>Type: scalar index<br/>In: current step<br/>Out: [1]"] --> S

    subgraph S["Reverse DDPM Sampling Loop x 1000 timesteps"]
        X["GraphDenoiser Input<br/>Type: tuple (x_t, t, cond)<br/>In: [1, T, 263], [1], [1, 512]<br/>Out: [1, T, 263], [1], [1, 512]"] --> D1["Linear Input Projection<br/>In: [1, T, 263]<br/>Out: [1, T, 1408]"]
        X --> TE1["Sinusoidal Timestep Embedding<br/>In: [1]<br/>Out: [1, 256]"]
        TE1 --> TE2["Linear + SiLU<br/>In: [1, 256]<br/>Out: [1, 1408]"]
        TE2 --> TE3["Linear<br/>In: [1, 1408]<br/>Out: [1, 1408]"]
        X --> CE1["Linear Condition Projection<br/>In: [1, 512]<br/>Out: [1, 1408]"]
        D1 --> ADD["Broadcast Add<br/>In: [1, T, 1408] + [1, 1408] + [1, 1408]<br/>Out: [1, T, 1408]"]
        TE3 --> ADD
        CE1 --> ADD
        ADD --> R1["Reshape To Nodes<br/>In: [1, T, 1408]<br/>Out: [T, 22, 64]"]
        R1 --> G1["GraphConv Block 1<br/>Linear + A_norm matmul + SiLU + LayerNorm + Residual<br/>In: [T, 22, 64]<br/>Out: [T, 22, 64]"]
        G1 --> G2["GraphConv Block 2<br/>Linear + A_norm matmul + SiLU + LayerNorm + Residual<br/>In: [T, 22, 64]<br/>Out: [T, 22, 64]"]
        G2 --> G3["GraphConv Block 3<br/>Linear + A_norm matmul + SiLU + LayerNorm + Residual<br/>In: [T, 22, 64]<br/>Out: [T, 22, 64]"]
        G3 --> G4["GraphConv Block 4<br/>Linear + A_norm matmul + SiLU + LayerNorm + Residual<br/>In: [T, 22, 64]<br/>Out: [T, 22, 64]"]
        G4 --> R2["Reshape Back<br/>In: [T, 22, 64]<br/>Out: [1, T, 1408]"]
        R2 --> O1["Linear Output Projection<br/>In: [1, T, 1408]<br/>Out: [1, T, 263] predicted noise"]
    end

    O1 --> M["Final Motion Output<br/>Type: generated motion features<br/>In: [1, T, 263]<br/>Out: [1, T, 263]"]
```
