import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "The framework to build agentic web apps",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
