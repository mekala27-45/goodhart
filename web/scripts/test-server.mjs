// A tiny server for the static export, built to behave like the hosts the site runs on at their
// least helpful: it refuses HEAD with a 405, and it answers its very first request with a 503 and a
// Retry-After, the way a host answers while it wakes. The site's fetch helper has to retry the 503
// and must never send HEAD; the end to end tests check both.
//
//   node scripts/test-server.mjs [--port 4173] [--root out] [--data dir] [--base /goodhart] [--log]
//
//   --port 0      pick a free port; the first line printed is JSON with the port
//   --data dir    files here are served in place of out/data (the end to end fixtures)
//   --log         print one JSON line per request: method, path, status
import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function args(argv) {
  const o = { port: 4173, root: join(web, "out"), data: null, base: "/goodhart", log: false };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (flag === "--log") o.log = true;
    else if (flag === "--port") o.port = Number(argv[++i]);
    else if (flag === "--root") o.root = resolve(argv[++i]);
    else if (flag === "--data") o.data = resolve(argv[++i]);
    else if (flag === "--base") o.base = argv[++i].replace(/\/$/, "");
    else throw new Error(`unknown argument ${flag}`);
  }
  return o;
}

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".jsonl": "application/x-ndjson; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
  ".wasm": "application/wasm",
  ".parquet": "application/vnd.apache.parquet",
};

export function serve(options) {
  const o = { ...args([]), ...options };
  if (!existsSync(join(o.root, "index.html"))) throw new Error(`${o.root}/index.html is missing: run npm run build first`);
  let first = true;
  const log = (req, status) => {
    if (o.log) process.stdout.write(`${JSON.stringify({ method: req.method, path: req.url, status })}\n`);
  };
  const reply = (req, res, status, headers = {}, body = "") => {
    res.writeHead(status, headers);
    res.end(body);
    log(req, status);
  };

  const server = createServer((req, res) => {
    // The host refuses HEAD, whatever the path; a client that sends one fails here.
    if (req.method === "HEAD") return reply(req, res, 405, { Allow: "GET" });
    // The very first request of the server's life is answered as a waking host answers it.
    if (first) {
      first = false;
      return reply(req, res, 503, { "Content-Type": "text/plain; charset=utf-8", "Retry-After": "1" }, "the host is waking; ask again");
    }
    if (req.method !== "GET") return reply(req, res, 405, { Allow: "GET" });

    const url = new URL(req.url ?? "/", "http://localhost");
    let path = decodeURIComponent(url.pathname);
    if (path === "/" || path === o.base) return reply(req, res, 301, { Location: `${o.base}/` });
    if (!path.startsWith(`${o.base}/`)) return notFound(req, res);
    path = path.slice(o.base.length);

    let file = null;
    if (o.data && path.startsWith("/data/")) {
      const candidate = normalize(join(o.data, path.slice("/data/".length)));
      if (candidate.startsWith(o.data + sep) && existsSync(candidate) && statSync(candidate).isFile()) file = candidate;
    }
    if (!file) {
      file = normalize(join(o.root, path));
      if (file !== o.root && !file.startsWith(o.root + sep)) return notFound(req, res);
      if (existsSync(file) && statSync(file).isDirectory()) {
        if (!url.pathname.endsWith("/")) return reply(req, res, 301, { Location: `${url.pathname}/${url.search}` });
        file = join(file, "index.html");
      }
      if (!existsSync(file) && existsSync(`${file}.html`)) file = `${file}.html`;
    }
    if (!existsSync(file) || !statSync(file).isFile()) return notFound(req, res);

    const size = statSync(file).size;
    const headers = { "Content-Type": TYPES[extname(file)] ?? "application/octet-stream", "Accept-Ranges": "bytes", "Cache-Control": "no-cache" };
    const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? "");
    if (range && (range[1] || range[2])) {
      let start = range[1] ? Number(range[1]) : Math.max(size - Number(range[2]), 0);
      const end = range[1] && range[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
      if (start >= size || start > end) return reply(req, res, 416, { ...headers, "Content-Range": `bytes */${size}` });
      start = Math.max(start, 0);
      res.writeHead(206, { ...headers, "Content-Range": `bytes ${start}-${end}/${size}`, "Content-Length": end - start + 1 });
      createReadStream(file, { start, end }).pipe(res);
      return log(req, 206);
    }
    res.writeHead(200, { ...headers, "Content-Length": size });
    createReadStream(file).pipe(res);
    return log(req, 200);
  });

  function notFound(req, res) {
    const page = join(o.root, "404.html");
    res.writeHead(404, { "Content-Type": TYPES[".html"] });
    if (existsSync(page)) createReadStream(page).pipe(res);
    else res.end();
    log(req, 404);
  }

  return new Promise((ok) => {
    server.listen(o.port, "127.0.0.1", () => {
      const port = server.address().port;
      ok({ server, port });
    });
  });
}

const invoked = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (invoked) {
  const o = args(process.argv.slice(2));
  const { port } = await serve(o);
  process.stdout.write(`${JSON.stringify({ listening: `http://127.0.0.1:${port}${o.base}/`, port })}\n`);
}
