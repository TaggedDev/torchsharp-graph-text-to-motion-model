using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

var processedPath = Path.GetFullPath(
    Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "processed"));

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

        foreach (var file in Directory.EnumerateFiles(dir, "*.bin").OrderBy(f => f))
        {
            var id = Path.GetFileNameWithoutExtension(file);
            var (frameCount, _, caption) = ReadBinHeader(file);
            result.Add(new { id, split, frameCount, caption });
        }
    }
    return Results.Json(result);
});

app.MapGet("/api/animation/{split}/{id}", (string split, string id) =>
{
    var file = Path.Combine(processedPath, split, $"{id}.bin");
    if (!File.Exists(file))
        return Results.NotFound();

    var (frameCount, featureDim, caption) = ReadBinHeader(file);
    var motionData = ReadBinFrames(file, frameCount, featureDim);

    // Extract joint positions: indices [4:67] = 21 joints x 3
    // Plus root height from index [3] for the pelvis (joint 0)
    var positions = new float[frameCount][];
    for (int f = 0; f < frameCount; f++)
    {
        // 22 joints x 3 = 66 floats per frame
        var frame = new float[66];

        // Joint 0 (pelvis): x=0, y=rootHeight, z=0
        frame[0] = 0f;
        frame[1] = motionData[f * featureDim + 3]; // root height Y
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
});

// Fallback to index.html for SPA
app.MapFallbackToFile("index.html");

Console.WriteLine($"Data path: {processedPath}");
app.Run();

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
