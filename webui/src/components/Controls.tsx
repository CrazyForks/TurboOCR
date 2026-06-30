import { useRef } from "react";
import { Upload, MousePointer2 } from "lucide-react";
import { Button, PillToggle, LayerChip, Switch, Badge, Spinner } from "./ui";
import { ExportMenu } from "./ExportMenu";
import { colorForClass, LINE_COLOR } from "../lib/colors";
import type { Layers } from "./viewer/types";
import type { Capabilities, RunOptions } from "../lib/types";

const REGION_COLOR = colorForClass(22).stroke;
const TABLE_COLOR = colorForClass(21).stroke;
const FORMULA_COLOR = colorForClass(5).stroke;

export interface Aggregate {
  layout: boolean;
  tables: boolean;
  formulas: boolean;
  done: number;
  running: number;
  total: number;
  degraded: string[];
}

export function Controls({
  onFile,
  fileName,
  caps,
  opts,
  setOpts,
  layers,
  setLayers,
  selectText,
  setSelectText,
  agg,
  onExportPdf,
  onExportText,
  onExportJson,
  exporting,
  hasDoc,
}: {
  onFile: (f: File) => void;
  fileName: string | null;
  caps: Capabilities | null;
  opts: RunOptions;
  setOpts: (o: RunOptions) => void;
  layers: Layers;
  setLayers: (l: Layers) => void;
  selectText: boolean;
  setSelectText: (v: boolean) => void;
  agg: Aggregate;
  onExportPdf: () => void;
  onExportText: () => void;
  onExportJson: () => void;
  exporting: boolean;
  hasDoc: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const tablesOk = caps?.features.tables ?? false;
  const formulasOk = caps?.features.formulas ?? false;
  const layoutOk = caps?.features.layout ?? false;

  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-b border-neutral-200 px-5 py-2.5">
      <Button onClick={() => inputRef.current?.click()}>
        <Upload className="h-4 w-4" />
        {fileName ? "Replace" : "Upload"}
        <input
          ref={inputRef}
          type="file"
          accept="image/*,application/pdf"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onFile(f);
            e.target.value = "";
          }}
        />
      </Button>
      {fileName && (
        <span className="max-w-[14rem] truncate text-sm text-neutral-500">{fileName}</span>
      )}

      <div className="flex items-center gap-1.5">
        <span className="mr-1 text-xs font-medium uppercase tracking-wide text-neutral-400">
          Stages
        </span>
        <PillToggle
          label="Layout"
          active={opts.layout}
          disabled={!layoutOk}
          title={layoutOk ? undefined : "No layout model loaded"}
          onClick={() => setOpts({ ...opts, layout: !opts.layout })}
        />
        <PillToggle
          label="Tables"
          active={opts.tables}
          disabled={!tablesOk}
          title={tablesOk ? undefined : "Start server with TABLE_BACKEND=slanext"}
          onClick={() => setOpts({ ...opts, tables: !opts.tables, layout: !opts.tables || opts.layout })}
        />
        <PillToggle
          label="Formulas"
          active={opts.formulas}
          disabled={!formulasOk}
          title={formulasOk ? undefined : "Start server with FORMULA_BACKEND=ppformulanet_s"}
          onClick={() =>
            setOpts({ ...opts, formulas: !opts.formulas, layout: !opts.formulas || opts.layout })
          }
        />
      </div>

      <div className="ml-auto flex items-center gap-3">
        {agg.running > 0 && (
          <span className="flex items-center gap-1.5 text-xs text-neutral-500">
            <Spinner /> OCR {agg.done}/{agg.total}
          </span>
        )}
        {agg.running === 0 && agg.total > 1 && (
          <Badge tone="green">
            {agg.done}/{agg.total} pages
          </Badge>
        )}
        {caps && <Badge tone="indigo">{String(caps.build).toUpperCase()}</Badge>}
        {agg.degraded.map((d) => (
          <Badge key={d} tone="amber">
            {d} degraded
          </Badge>
        ))}

        <span className="h-5 w-px bg-neutral-200" />

        <div className="flex items-center gap-1.5">
          <LayerChip
            label="Text"
            color={LINE_COLOR.stroke}
            active={layers.lines}
            onClick={() => setLayers({ ...layers, lines: !layers.lines })}
          />
          <LayerChip
            label="Regions"
            color={REGION_COLOR}
            active={layers.layout}
            disabled={!agg.layout}
            onClick={() => setLayers({ ...layers, layout: !layers.layout })}
          />
          <LayerChip
            label="Tables"
            color={TABLE_COLOR}
            active={layers.tables}
            disabled={!agg.tables}
            onClick={() => setLayers({ ...layers, tables: !layers.tables })}
          />
          <LayerChip
            label="Formulas"
            color={FORMULA_COLOR}
            active={layers.formulas}
            disabled={!agg.formulas}
            onClick={() => setLayers({ ...layers, formulas: !layers.formulas })}
          />
        </div>

        <label className="flex items-center gap-1.5 text-sm text-neutral-600">
          <MousePointer2 className="h-4 w-4 text-neutral-400" />
          Select text
          <Switch checked={selectText} onChange={setSelectText} />
        </label>

        <ExportMenu
          onPdf={onExportPdf}
          onText={onExportText}
          onJson={onExportJson}
          busy={exporting}
          disabled={!hasDoc}
        />
      </div>
    </div>
  );
}
