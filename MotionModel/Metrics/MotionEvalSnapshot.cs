namespace Text2Motion.TorchTrainer;

public sealed record MotionEvalSnapshot(
    int Epoch,
    MotionEvalPhase Phase,
    float L2Distance);
