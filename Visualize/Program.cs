using System.Collections.Concurrent;
using System.Text;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

var solutionRoot = Path.GetFullPath(
    Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));
var processedPath = Path.Combine(solutionRoot, "processed");
var checkpointsPath = Path.Combine(solutionRoot, "checkpoints");
var generationsPath = Path.Combine(solutionRoot, "generations");
var trainingCfg = Path.Combine(solutionRoot, "Configs", "training.json");
var preprocessingCfg = Path.Combine(solutionRoot, "Configs", "preprocessing.json");
var meanPath = Path.Combine(processedPath, "mean.bin");
var stdPath = Path.Combine(processedPath, "std.bin");

// Engine cache: one per checkpoint. GPU work is serialized by the semaphore.
var engineCache = new ConcurrentDictionary<string, InferenceEngine>();
var engineLock = new SemaphoreSlim(1, 1);

// Skeleton edges from Data/Skeleton.cs (parent array)
int[] parent = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19];
var edges = parent
    .Select((p, c) => new { p, c })
    .Where(e => e.p >= 0)
    .Select(e => new int[] { e.p, e.c })
    .ToArray();

string[] jointNames =
[
    "pelvis", "right_hip", "left_hip", "spine1", "right_knee", "left_knee",
    "spine2", "right_ankle", "left_ankle", "spine3", "right_foot", "left_foot",
    "neck", "right_collar", "left_collar", "head", "right_shoulder", "left_shoulder",
    "right_elbow", "left_elbow", "right_wrist", "left_wrist"
];

// Joint group colors (CSS color names)
var jointGroup = new string[22];
int[] leftLeg = [2, 5, 8, 11];
int[] rightLeg = [1, 4, 7, 10];
int[] spine = [0, 3, 6, 9, 12, 15];
int[] leftArm = [14, 17, 19, 21];
int[] rightArm = [13, 16, 18, 20];
foreach (var i in leftLeg) jointGroup[i] = "left_leg";
foreach (var i in rightLeg) jointGroup[i] = "right_leg";
foreach (var i in spine) jointGroup[i] = "spine";
foreach (var i in leftArm) jointGroup[i] = "left_arm";
foreach (var i in rightArm) jointGroup[i] = "right_arm";

app.UseStaticFiles();

app.MapGet("/api/animations", () =>
{
    var result = new List<object>();
    foreach (var split in new[] { "train", "val", "test" })
    {
        var dir = Path.Combine(processedPath, split);
        if (!Directory.Exists(dir)) continue;

        foreach (var file in Directory.EnumerateFiles(dir, "*.bin")
            .Where(f => !f.EndsWith(".clip.bin", StringComparison.OrdinalIgnoreCase))
            .OrderBy(f => f))
        {
            var id = Path.GetFileNameWithoutExtension(file);
            var (frameCount, _, caption) = ReadBinHeader(file);
            result.Add(new { id, split, frameCount, caption });
        }
    }

    // Generated files live under generations/{modelName}/*.bin
    if (Directory.Exists(generationsPath))
    {
        foreach (var modelDir in Directory.EnumerateDirectories(generationsPath))
        {
            var modelName = Path.GetFileName(modelDir);
            var splitLabel = $"generations/{modelName}";
            foreach (var file in Directory.EnumerateFiles(modelDir, "*.bin")
                .OrderByDescending(File.GetLastWriteTime))
            {
                var id = Path.GetFileNameWithoutExtension(file);
                var (frameCount, _, caption) = ReadBinHeader(file);
                result.Add(new { id, split = splitLabel, frameCount, caption });
            }
        }
    }

    return Results.Json(result);
});

// Catch-all so split can be nested like "generations/model_epoch50_..."
app.MapGet("/api/animation/{**path}", (string path) =>
{
    var lastSlash = path.LastIndexOf('/');
    if (lastSlash < 0) return Results.NotFound();
    var split = path[..lastSlash];
    var id = path[(lastSlash + 1)..];

    string file;
    if (split.StartsWith("generations/", StringComparison.Ordinal))
    {
        var modelName = split["generations/".Length..];
        file = Path.Combine(generationsPath, modelName, $"{id}.bin");
    }
    else
    {
        file = Path.Combine(processedPath, split, $"{id}.bin");
    }

    // Path traversal guard
    var fullFile = Path.GetFullPath(file);
    if (!fullFile.StartsWith(Path.GetFullPath(processedPath), StringComparison.OrdinalIgnoreCase)
        && !fullFile.StartsWith(Path.GetFullPath(generationsPath), StringComparison.OrdinalIgnoreCase))
    {
        return Results.NotFound();
    }

    if (!File.Exists(fullFile))
        return Results.NotFound();

    return BuildAnimationResponse(fullFile);
});

// List available checkpoints
app.MapGet("/api/models", () =>
{
    if (!Directory.Exists(checkpointsPath))
        return Results.Json(Array.Empty<object>());

    var items = Directory.EnumerateFiles(checkpointsPath, "*.pt")
        .OrderByDescending(File.GetLastWriteTime)
        .Select(f => new
        {
            name = Path.GetFileNameWithoutExtension(f),
            file = Path.GetFileName(f)
        })
        .ToArray();
    return Results.Json(items);
});

// Run inference
app.MapPost("/api/generate", async (GenerateRequest req) =>
{
    if (string.IsNullOrWhiteSpace(req.Model))
        return Results.BadRequest(new { error = "Model is required" });
    if (string.IsNullOrWhiteSpace(req.Prompt))
        return Results.BadRequest(new { error = "Prompt is required" });

    int frames = req.Frames <= 0 ? 120 : req.Frames;
    frames = Math.Clamp(frames, 1, 300);

    var modelFile = req.Model.EndsWith(".pt", StringComparison.OrdinalIgnoreCase)
        ? req.Model
        : req.Model + ".pt";
    var ckpt = Path.Combine(checkpointsPath, modelFile);
    if (!File.Exists(ckpt))
        return Results.NotFound(new { error = "Checkpoint not found" });

    await engineLock.WaitAsync();
    try
    {
        InferenceEngine engine;
        try
        {
            engine = engineCache.GetOrAdd(ckpt, p => new InferenceEngine(
                p, trainingCfg, preprocessingCfg, meanPath, stdPath));
        }
        catch (Exception ex)
        {
            return Results.Problem(
                detail: $"Failed to load engine: {ex.Message}",
                statusCode: 500);
        }

        string binPath;
        try
        {
            binPath = engine.Generate(req.Prompt, frames, generationsPath);
        }
        catch (Exception ex)
        {
            return Results.Problem(
                detail: $"Generation failed: {ex.Message}",
                statusCode: 500);
        }

        var modelName = Path.GetFileNameWithoutExtension(ckpt);
        var id = Path.GetFileNameWithoutExtension(binPath);
        var (frameCount, _, caption) = ReadBinHeader(binPath);
        return Results.Json(new
        {
            split = $"generations/{modelName}",
            id,
            frameCount,
            caption
        });
    }
    finally
    {
        engineLock.Release();
    }
});

// Fallback to index.html for SPA
app.MapFallbackToFile("index.html");

Console.WriteLine($"Solution root: {solutionRoot}");
Console.WriteLine($"Data path:     {processedPath}");
Console.WriteLine($"Checkpoints:   {checkpointsPath}");
Console.WriteLine($"Generations:   {generationsPath}");
app.Run();


IResult BuildAnimationResponse(string file)
{
    var (frameCount, featureDim, caption) = ReadBinHeader(file);
    var motionData = ReadBinFrames(file, frameCount, featureDim);

    if (!TryFindFirstNonFinite(motionData, out var badIndex, out var badValue))
    {
        int badFrame = badIndex / featureDim;
        int badFeature = badIndex % featureDim;
        return Results.Problem(
            detail: $"Animation file '{Path.GetFileName(file)}' contains a non-finite value ({badValue}) at frame {badFrame}, feature {badFeature}. Regenerate it with a stable checkpoint.",
            statusCode: 422);
    }

    // Extract joint positions: indices [4:67] = 21 joints x 3
    // Plus root height from index [3] for the pelvis (joint 0)
    var positions = new float[frameCount][];
    for (int f = 0; f < frameCount; f++)
    {
        var frame = new float[66];

        // Joint 0 (pelvis): x=0, y=rootHeight, z=0
        frame[0] = 0f;
        frame[1] = motionData[f * featureDim + 3];
        frame[2] = 0f;

        // Joints 1-21: copy from [4:67]
        Array.Copy(motionData, f * featureDim + 4, frame, 3, 63);

        positions[f] = frame;
    }

    return Results.Json(new
    {
        caption,
        frameCount,
        joints = 22,
        edges,
        jointNames,
        jointGroup,
        positions
    });
}

static (int frameCount, int featureDim, string caption) ReadBinHeader(string path)
{
    using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
    using var br = new BinaryReader(fs);

    var frameCount = br.ReadInt32();
    var featureDim = br.ReadInt32();

    // Skip motion data to read caption
    fs.Seek((long)frameCount * featureDim * sizeof(float), SeekOrigin.Current);

    var captionLen = br.ReadInt32();
    var captionBytes = br.ReadBytes(captionLen);
    var caption = Encoding.UTF8.GetString(captionBytes);

    return (frameCount, featureDim, caption);
}

static float[] ReadBinFrames(string path, int frameCount, int featureDim)
{
    using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
    using var br = new BinaryReader(fs);

    br.ReadInt32(); // frameCount
    br.ReadInt32(); // featureDim

    var byteCount = frameCount * featureDim * sizeof(float);
    var bytes = br.ReadBytes(byteCount);
    var floats = new float[frameCount * featureDim];
    Buffer.BlockCopy(bytes, 0, floats, 0, byteCount);
    return floats;
}

static bool TryFindFirstNonFinite(float[] values, out int index, out float value)
{
    for (int i = 0; i < values.Length; i++)
    {
        var current = values[i];
        if (float.IsNaN(current) || float.IsInfinity(current))
        {
            index = i;
            value = current;
            return false;
        }
    }

    index = -1;
    value = 0f;
    return true;
}

record GenerateRequest(string Model, string Prompt, int Frames);
