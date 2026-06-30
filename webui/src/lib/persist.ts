import type { OcrResponse, RunOptions } from "./types";

// Persist the open document AND its OCR results in IndexedDB so a page reload
// restores the full state — no re-render, no re-OCR. IndexedDB structured-clones
// File/Blob and ArrayBuffer directly (localStorage can't hold binary).
//
// Two shapes share one key: a lightweight {full:false} record written the moment
// a file is opened (so a reload mid-processing at least restores the document and
// re-runs OCR), upgraded to a {full:true} snapshot once every page is done.
const DB_NAME = "turboocr-studio";
const STORE = "kv";
const KEY = "doc";

export interface RasterRec {
  id: number;
  bytes: ArrayBuffer;
  type: string;
}
export interface ResultRec {
  id: number;
  response: OcrResponse | null;
  elapsedMs?: number;
}
export interface PageRec {
  id: number;
  w: number;
  h: number;
}
export interface Snapshot {
  full: true;
  file: Blob;
  name: string;
  kind: "image" | "pdf";
  opts: RunOptions;
  pages: PageRec[];
  rasters: RasterRec[];
  results: ResultRec[];
}
interface FileOnly {
  full: false;
  file: Blob;
  name: string;
}
type Persisted = Snapshot | FileOnly;

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open(DB_NAME, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

function reqAsync<T>(r: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

async function put(value: Persisted): Promise<void> {
  try {
    const db = await openDb();
    const store = db.transaction(STORE, "readwrite").objectStore(STORE);
    await reqAsync(store.put(value, KEY));
  } catch {
    /* private mode / quota — ignore */
  }
}

// Written immediately on open: enough to restore + re-OCR after a reload that
// happens before processing finishes.
export function saveFile(file: File): Promise<void> {
  return put({ full: false, file, name: file.name });
}

// Written when every page is done: restores the whole state with no re-OCR.
export function saveSnapshot(snap: Omit<Snapshot, "full">): Promise<void> {
  return put({ full: true, ...snap });
}

export async function loadState(): Promise<Persisted | null> {
  try {
    const db = await openDb();
    const store = db.transaction(STORE, "readonly").objectStore(STORE);
    const v = await reqAsync<Persisted | undefined>(store.get(KEY));
    return v && v.file ? v : null;
  } catch {
    return null;
  }
}

export async function clearState(): Promise<void> {
  try {
    const db = await openDb();
    const store = db.transaction(STORE, "readwrite").objectStore(STORE);
    await reqAsync(store.delete(KEY));
  } catch {
    /* ignore */
  }
}
