import { DocsLayout } from "fumadocs-ui/layouts/docs";
import type { Metadata } from "next";
import type { ReactNode } from "react";
import { baseOptions } from "@/app/docs/layout.config";
import { source } from "@/lib/source";

export const metadata: Metadata = {
  title: "Documentation",
};

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <DocsLayout tree={source.pageTree} {...baseOptions}>
      {children}
    </DocsLayout>
  );
}
