import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Deliberation Map Simulation",
  description: "Multi-agent AI deliberation map research simulator",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ background: "#060d1a", color: "#e2e8f0", fontFamily: "system-ui, sans-serif", margin: 0 }}>
        <nav style={{ borderBottom: "1px solid rgba(255,255,255,0.08)", padding: "12px 24px", display: "flex", gap: "24px", alignItems: "center" }}>
          <span style={{ color: "#00d4ff", fontWeight: 700, fontSize: "16px" }}>
            DeliberationSim
          </span>
          <a href="/" style={{ color: "#94a3b8", textDecoration: "none", fontSize: "14px" }}>Map</a>
          <a href="/simulation" style={{ color: "#94a3b8", textDecoration: "none", fontSize: "14px" }}>Simulation</a>
          <a href="/analytics" style={{ color: "#94a3b8", textDecoration: "none", fontSize: "14px" }}>Analytics</a>
          <a href="/user-models" style={{ color: "#94a3b8", textDecoration: "none", fontSize: "14px" }}>User Models</a>
        </nav>
        {children}
      </body>
    </html>
  );
}
