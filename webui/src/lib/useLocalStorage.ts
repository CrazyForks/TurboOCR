import { useEffect, useState } from "react";

// Persists a small piece of UI state (toggles, view prefs) across reloads.
export function useLocalStorage<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw !== null ? (JSON.parse(raw) as T) : initial;
    } catch {
      return initial;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* private mode / quota — ignore */
    }
  }, [key, value]);

  return [value, setValue] as const;
}
