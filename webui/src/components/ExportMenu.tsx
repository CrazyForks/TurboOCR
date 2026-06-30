import { useEffect, useRef, useState } from "react";
import { Download, ChevronDown } from "lucide-react";
import { Button, Spinner } from "./ui";

export function ExportMenu({
  onPdf,
  onText,
  onJson,
  busy,
  disabled,
}: {
  onPdf: () => void;
  onText: () => void;
  onJson: () => void;
  busy: boolean;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  const item =
    "block w-full px-3 py-1.5 text-left text-sm text-neutral-700 hover:bg-neutral-100";
  const pick = (fn: () => void) => {
    setOpen(false);
    fn();
  };

  return (
    <div ref={ref} className="relative">
      <Button disabled={disabled} onClick={() => setOpen((o) => !o)}>
        {busy ? <Spinner /> : <Download className="h-4 w-4" />}
        Download
        <ChevronDown className="h-3.5 w-3.5" />
      </Button>
      {open && (
        <div className="absolute right-0 z-20 mt-1 w-44 overflow-hidden rounded-lg border border-neutral-200 bg-white py-1 shadow-lg">
          <button className={item} onClick={() => pick(onPdf)}>
            Searchable PDF
          </button>
          <button className={item} onClick={() => pick(onText)}>
            Plain text (.txt)
          </button>
          <button className={item} onClick={() => pick(onJson)}>
            JSON
          </button>
        </div>
      )}
    </div>
  );
}
