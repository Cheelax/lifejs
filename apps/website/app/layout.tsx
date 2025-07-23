import { WebsiteHeader } from "@/components/website-header";
import "./global.css";
import { GoogleAnalytics } from "@next/third-parties/google";
import { RootProvider } from "fumadocs-ui/provider";
import { GeistMono } from "geist/font/mono";
import type { Metadata } from "next";
import { Gloria_Hallelujah } from "next/font/google";
import localFont from "next/font/local";
import type { ReactNode } from "react";

/* Fonts */

const sfProDisplay = localFont({
  src: "../public/fonts/sfprodisplay-medium.otf",
  variable: "--font-sf-pro-display",
});

const ppNeueMontreal = localFont({
  src: [
    {
      path: "../public/fonts/ppneuemontreal-regular.woff2",
      weight: "400",
      style: "normal",
    },
    {
      path: "../public/fonts/ppneuemontreal-medium.otf",
      weight: "500",
      style: "normal",
    },
    {
      path: "../public/fonts/ppneuemontreal-italic.otf",
      weight: "500",
      style: "italic",
    },
  ],
  variable: "--font-pp-neue-montreal",
});

const gloriaHallelujah = Gloria_Hallelujah({
  subsets: ["latin"],
  weight: "400",
  variable: "--font-gloria-hallelujah",
});

/* Metadata */

const name = "Life.js";
const description =
  "Life.js is the first fullstack framework to build agentic web applications. It is minimal, extensible, and typesafe. Well, everything you love.";
const url = "https://lifejs.org";

export const viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export const metadata: Metadata = {
  metadataBase: new URL(url),
  applicationName: name,
  title: {
    template: `${name} • %s`,
    default: "Untitled Page",
  },
  description,
  keywords: [
    "react",
    "framework",
    "typescript",
    "fullstack",
    "agents",
    "agentic",
    "life",
    "nextjs",
  ],
  openGraph: {
    title: `${name}`,
    description,
    siteName: name,
    locale: "en_US",
    type: "website",
    url,
  },
  twitter: {
    card: "summary_large_image",
    site: "@lifejs_org",
    creator: "@lifejs_org",
    title: `${name}`,
    description,
  },
};

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html
      className={`${sfProDisplay.variable} ${ppNeueMontreal.variable} ${GeistMono.variable} ${gloriaHallelujah.variable}`}
      lang="en"
      suppressHydrationWarning
    >
      <body className="flex min-h-screen flex-col bg-gradient-to-b from-white to-[#FEF7F2]">
        <RootProvider>
          <WebsiteHeader />
          {children}
        </RootProvider>
        <GoogleAnalytics gaId="G-SLGSBMQZ3J" />
      </body>
    </html>
  );
}
