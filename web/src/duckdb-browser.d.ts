// The browser build of DuckDB-WASM, imported by path so the server pass of the static export never
// resolves the node build; its types are the package's own.
declare module "@duckdb/duckdb-wasm/dist/duckdb-browser.mjs" {
  export * from "@duckdb/duckdb-wasm";
}
