import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",          // static HTML/JS/CSS — no Node server needed
  trailingSlash: true,       // /setup → /setup/index.html
  images: { unoptimized: true },
};

export default nextConfig;
