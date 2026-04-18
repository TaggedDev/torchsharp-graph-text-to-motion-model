namespace Text2Motion.TorchTrainer;

public sealed record MotionEvalSnapshot(
    int Epoch,
    MotionEvalPhase Phase,
    float RPrecisionTop1,
    float RPrecisionTop2,
    float RPrecisionTop3,
    float FrechetInceptionDistance,
    float MultimodalDistance,
    float Diversity,
    float Multimodality);
