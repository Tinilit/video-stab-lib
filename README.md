
# video-stab-lib

This is a native C++ dynamic library that wraps OpenCV's [`cv::videostab`](https://docs.opencv.org/4.x/dc/dc3/group__videostab.html) module to perform video stabilization, and exposes a minimal C-style API that can be used from .NET (e.g., C#, WPF, or Unity).

The library allows feeding raw video frames (RGBA format), performs one-pass stabilization using OpenCV's internal algorithms, and returns the stabilized frames back to your application.

## ✅ Features

- Uses OpenCV's `OnePassStabilizer` internally
- Accepts input frames as `unsigned char*` (RGBA)
- Outputs stabilized frames in the same format
- Designed for easy integration with SkiaSharp (`SKBitmap.GetPixels()`)
- Compatible with .NET via `DllImport`

## 🔧 Use Cases

- Stabilize webcam or recorded video in .NET apps
- Integrate video stabilization in WPF projects
- Pre-process frames for machine learning inference

## 📦 Example Integration (C#)

```csharp
[DllImport("video_stab_lib.dll")]
static extern IntPtr CreateStabilizer();

[DllImport("video_stab_lib.dll")]
static extern void FeedFrame(IntPtr stab, IntPtr data, int width, int height, int stride);

[DllImport("video_stab_lib.dll")]
static extern void ProcessStabilization(IntPtr stab);

[DllImport("video_stab_lib.dll")]
static extern bool GetFrame(IntPtr stab, int index, IntPtr outBuffer, int width, int height, int stride);

[DllImport("video_stab_lib.dll")]
static extern void FreeStabilizer(IntPtr stab);
