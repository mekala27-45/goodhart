// Reads a grades parquet file (zstd, as goodhart_evaluation.grade writes it) into plain rows and into
// an Arrow IPC stream for DuckDB-WASM. Why not read_parquet in the engine: the 1.32.0 bundle links only
// DuckDB's core, and its first read_parquet downloads the parquet extension from
// extensions.duckdb.org; this site fetches nothing from anywhere but its own origin. So the file is
// fetched whole with the site's GET helper, decoded here by hyparquet with fzstd, and the engine runs
// the SQL over the table it is handed. The SQL is the same SQL.
import { decompress } from "fzstd";
import { parquetMetadata, parquetReadObjects } from "hyparquet";

export type Cell = string | number | boolean | null;
export type ColumnType = "string" | "int" | "double" | "bool";

export interface Decoded {
  columns: { name: string; type: ColumnType }[];
  rows: Record<string, Cell>[];
}

const compressors = {
  ZSTD: (input: Uint8Array, length: number) => decompress(input, new Uint8Array(length)),
};

function columnType(type: string | undefined, logical: string | undefined): ColumnType {
  if (type === "BOOLEAN") return "bool";
  if (type === "INT32" || type === "INT64") return logical === "DECIMAL" ? "double" : "int";
  if (type === "FLOAT" || type === "DOUBLE") return "double";
  return "string";
}

function cell(value: unknown, type: ColumnType): Cell {
  if (value === null || value === undefined) return null;
  switch (type) {
    case "bool":
      return Boolean(value);
    case "int":
    case "double":
      return typeof value === "bigint" ? Number(value) : Number(value);
    default:
      return typeof value === "string" ? value : String(value);
  }
}

/** Every row of a parquet file, typed by its schema: INT64 arrives as a number, never a BigInt. */
export async function decodeParquet(buffer: ArrayBuffer): Promise<Decoded> {
  const metadata = parquetMetadata(buffer);
  const leaves = metadata.schema.slice(1).filter((s) => !s.num_children);
  const columns = leaves.map((s) => ({ name: s.name, type: columnType(s.type, s.logical_type?.type) }));
  const raw = (await parquetReadObjects({ file: buffer, metadata, compressors })) as Record<string, unknown>[];
  const rows = raw.map((r) => Object.fromEntries(columns.map((c) => [c.name, cell(r[c.name], c.type)])));
  return { columns, rows };
}

/** The decoded rows as an Arrow IPC stream, column types kept, for insertArrowFromIPCStream. */
export async function toArrowStream(decoded: Decoded): Promise<Uint8Array> {
  const arrow = await import("apache-arrow");
  type AnyVector = InstanceType<typeof arrow.Vector>;
  const vectors: Record<string, AnyVector> = {};
  for (const c of decoded.columns) {
    const values = decoded.rows.map((r) => r[c.name] ?? null);
    const type =
      c.type === "bool" ? new arrow.Bool() : c.type === "int" ? new arrow.Int32() : c.type === "double" ? new arrow.Float64() : new arrow.Utf8();
    vectors[c.name] = arrow.vectorFromArray(values as never[], type as never) as unknown as AnyVector;
  }
  return arrow.tableToIPC(new arrow.Table(vectors), "stream");
}
