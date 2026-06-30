import { useEffect, useState } from "react";
import { Moon, Sun } from "lucide-react";

type Theme = "light" | "dark";

function getInitialTheme(): Theme {
  try {
    const saved = localStorage.getItem("turboocr-theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* private mode — ignore */
  }
  return typeof window !== "undefined" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    try {
      localStorage.setItem("turboocr-theme", theme);
    } catch {
      /* ignore */
    }
  }, [theme]);

  const dark = theme === "dark";
  return (
    <button
      type="button"
      aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
      title={dark ? "Light" : "Dark"}
      onClick={() => setTheme(dark ? "light" : "dark")}
      className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-neutral-200 bg-neutral-50 px-2.5 text-xs font-medium text-neutral-600 transition-colors hover:bg-neutral-100"
    >
      {dark ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
      {dark ? "Light" : "Dark"}
    </button>
  );
}