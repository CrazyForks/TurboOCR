import * as pdfjsLib from "pdfjs-dist";
// Vite bundles the worker and gives us its URL; API and worker versions match
// because both come from the same pinned pdfjs-dist.
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";

pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

export { pdfjsLib };
