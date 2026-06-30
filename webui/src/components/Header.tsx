import { ThemeToggle } from "./ThemeToggle";

// Swap these for the real sponsor URLs.
const SPONSORS = {
  miruiq: "https://miruiq.com",
  diaiq: "https://diaiq.com",
};

export function Header({ capsErr }: { capsErr: string | null }) {
  return (
    <header className="flex items-center gap-2 border-b border-neutral-200 bg-white px-5 py-3">
      <span className="flex h-[26px] w-[26px] items-center justify-center rounded-md bg-indigo-600 font-serif text-sm font-semibold text-[var(--accent-ink)]">
        T
      </span>
      <h1 className="font-serif text-[15px] font-semibold tracking-[-0.01em] text-neutral-900">
        TurboOCR
      </h1>
      <span className="font-mono text-[11px] uppercase tracking-wide text-neutral-400">
        Studio
      </span>
      {capsErr && (
        <span className="ml-3 text-xs text-red-600">backend unreachable: {capsErr}</span>
      )}

      <div className="ml-auto flex items-center gap-3">
        <ThemeToggle />
        <span className="text-xs text-neutral-400">
          Sponsored by{" "}
          <a
            href={SPONSORS.miruiq}
            target="_blank"
            rel="noopener noreferrer"
            className="font-medium text-neutral-500 underline-offset-2 hover:text-indigo-600 hover:underline"
          >
            Miruiq
          </a>{" "}
          ·{" "}
          <a
            href={SPONSORS.diaiq}
            target="_blank"
            rel="noopener noreferrer"
            className="font-medium text-neutral-500 underline-offset-2 hover:text-indigo-600 hover:underline"
          >
            Diaiq
          </a>
        </span>
      </div>
    </header>
  );
}
