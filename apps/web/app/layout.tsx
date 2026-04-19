import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { ThemeProvider } from "next-themes";
import { Toaster } from "sonner";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "LeverGuide — Decision Intelligence for Tabular Data",
  description:
    "Upload any dataset, pick a KPI, and get ranked, explainable recommendations " +
    "backed by predictive modelling and causal analysis.",
  keywords: ["causal inference", "ML", "decision intelligence", "KPI", "data analysis"],
  openGraph: {
    title: "LeverGuide",
    description: "Don't just predict. Decide what to change.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} font-sans antialiased`}>
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
          {children}
          <Toaster richColors position="top-right" />
        </ThemeProvider>
      </body>
    </html>
  );
}
