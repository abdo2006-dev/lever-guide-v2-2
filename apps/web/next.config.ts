import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow the frontend to proxy to the Python backend during local dev
  async rewrites() {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
    return [
      {
        source: "/api/analyze",
        destination: `${apiUrl}/api/analyze`,
      },
      {
        source: "/api/health",
        destination: `${apiUrl}/health`,
      },
    ];
  },
  // Strict mode helps surface issues early
  reactStrictMode: true,
};

export default nextConfig;
