// Small raster helpers shared by image ingest and PDF rendering. Everything is
// normalized to PNG so OCR input and searchable-PDF embedding are uniform.

export function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("image decode failed"));
    img.src = url;
  });
}

export async function canvasToPng(
  canvas: HTMLCanvasElement,
): Promise<{ bytes: ArrayBuffer; url: string }> {
  const blob: Blob = await new Promise((resolve, reject) =>
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error("canvas toBlob failed"))),
      "image/png",
    ),
  );
  return { bytes: await blob.arrayBuffer(), url: URL.createObjectURL(blob) };
}
