namespace Text2Motion.TorchTrainer;

public sealed record MotionEvalSnapshot(
    int Epoch,
    float FrechetInceptionDistance,
    float Diversity,
    float Multimodality,
    float RPrecisionTop1,
    float RPrecisionTop2,
    float RPrecisionTop3,
    float MultimodalDistance);
