import type { NextConfig } from "next";

// A static export for GitHub Pages, where the site lives at mekala27-45.github.io/goodhart. The base
// path is fixed: the Pages workflow, the test server and the Playwright tests all serve it there.
const basePath = "/goodhart";

// The grader API the desk posts to. A build may point it elsewhere (a local container while the API
// is developed); the deployed site always uses the Vercel address.
const api = process.env.NEXT_PUBLIC_GOODHART_API || "https://goodhart-grader.vercel.app";

const config: NextConfig = {
  output: "export",
  basePath,
  assetPrefix: basePath,
  trailingSlash: true,
  images: { unoptimized: true },
  reactStrictMode: true,
  poweredByHeader: false,
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
    NEXT_PUBLIC_GOODHART_API: api,
  },
  turbopack: { root: process.cwd() },
};

export default config;
