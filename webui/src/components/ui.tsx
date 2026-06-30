import { clsx } from "clsx";
import type { ButtonHTMLAttributes, ReactNode } from "react";

export function Button({
  className,
  variant = "default",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "ghost" | "primary";
}) {
  return (
    <button
      className={clsx(
        "inline-flex items-center justify-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40",
        variant === "primary" && "bg-indigo-600 text-white hover:bg-indigo-500",
        variant === "default" &&
          "border border-neutral-200 bg-white text-neutral-700 hover:bg-neutral-50",
        variant === "ghost" && "text-neutral-600 hover:bg-neutral-100",
        className,
      )}
      {...props}
    />
  );
}

// Pill toggle for pipeline stages — toggling re-runs OCR.
export function PillToggle({
  label,
  active,
  disabled,
  title,
  onClick,
}: {
  label: ReactNode;
  active: boolean;
  disabled?: boolean;
  title?: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      title={title}
      onClick={onClick}
      className={clsx(
        "rounded-full border px-3 py-1 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40",
        active
          ? "border-indigo-200 bg-indigo-50 text-indigo-700"
          : "border-neutral-200 bg-white text-neutral-500 hover:bg-neutral-50",
      )}
    >
      {label}
    </button>
  );
}

// Small colored chip for toggling an overlay layer's visibility.
export function LayerChip({
  label,
  color,
  active,
  disabled,
  onClick,
}: {
  label: string;
  color: string;
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-medium transition-colors disabled:opacity-30",
        active
          ? "border-indigo-200 bg-indigo-50 text-indigo-700"
          : "border-neutral-200 bg-white text-neutral-400 hover:bg-neutral-50",
      )}
    >
      <span
        className="inline-block h-2.5 w-2.5 rounded-sm"
        style={{ background: active ? color : "transparent", outline: `1px solid ${color}` }}
      />
      {label}
    </button>
  );
}

export function Switch({
  checked,
  onChange,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={clsx(
        "relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors disabled:opacity-40",
        checked ? "bg-indigo-600" : "bg-neutral-300",
      )}
    >
      <span
        className={clsx(
          "inline-block h-4 w-4 transform rounded-full bg-white shadow-sm transition-transform",
          checked ? "translate-x-4" : "translate-x-0.5",
        )}
      />
    </button>
  );
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "green" | "amber" | "red" | "indigo";
}) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium",
        tone === "neutral" && "bg-neutral-100 text-neutral-600",
        tone === "green" && "bg-ok/15 text-ok",
        tone === "amber" && "bg-mid/20 text-[oklch(0.5_0.12_80)] dark:text-[var(--mid)]",
        tone === "red" && "bg-low/15 text-low",
        tone === "indigo" && "bg-accent-soft text-accent",
      )}
    >
      {children}
    </span>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      className={clsx(
        "inline-block h-4 w-4 animate-spin rounded-full border-2 border-neutral-300 border-t-neutral-600",
        className,
      )}
    />
  );
}
