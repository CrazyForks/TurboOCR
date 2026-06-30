import { useMemo } from "react";
import katex from "katex";

export function KatexBlock({
  tex,
  display = true,
}: {
  tex: string;
  display?: boolean;
}) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(tex, {
        displayMode: display,
        throwOnError: false,
        strict: false,
      });
    } catch {
      return null;
    }
  }, [tex, display]);

  if (html === null) {
    return <code className="text-xs text-amber-300">{tex}</code>;
  }
  return (
    <div className="overflow-x-auto" dangerouslySetInnerHTML={{ __html: html }} />
  );
}
