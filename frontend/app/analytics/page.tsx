"use client";
import { useState, useEffect, useCallback } from "react";
import {
  getSessions,
  getRecommendationStats,
  getSkipRateTimeline,
  getInferenceAccuracy,
  getUserModelTree,
  trainUserModel,
  getSessionProfiles,
  getTreeGrowth,
  getAgentStats,
  getActivityPerTick,
} from "@/lib/api";
import { PTYPE_COLORS } from "@/components/MapTree/PostCard";

interface Session { id: string; status: string; tick_count: number; }
interface StrategyStats { strategy: string; total_recommendations: number; engaged: number; skipped: number; skip_rate: number; accuracy: number; }
interface Profile { agent_id: string; agent_name?: string; profile_text: string; last_updated_tick: number; }
interface GrowthPoint { tick: number; ptype: string; count: number; }
interface AgentStat { agent_id: string; agent_name: string; avatar_color: string; domain: string; total: number; by_ptype: Record<string, number>; }
interface ActivityPoint { tick: number; count: number; }

const STRATEGY_COLORS: Record<string, string> = {
  random: "#64748b",
  priority: "#10b981",
  multi_agent: "#00d4ff",
  ab_rotate: "#6366f1",
};

const card = {
  background: "rgba(255,255,255,0.03)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: "10px",
  padding: "20px",
} as const;

function SvgBarChart({ data, color = "#00d4ff", height = 64 }: {
  data: { label: string | number; value: number }[];
  color?: string;
  height?: number;
}) {
  if (data.length === 0) return <div style={{ color: "#334155", fontSize: "12px" }}>No data yet.</div>;
  const max = Math.max(...data.map((d) => d.value), 1);
  const barW = Math.max(4, Math.floor(560 / data.length) - 2);
  const gap = Math.max(1, Math.floor(560 / data.length) - barW);
  const totalW = data.length * (barW + gap);
  return (
    <svg width="100%" viewBox={`0 0 ${totalW} ${height + 16}`} preserveAspectRatio="none" style={{ display: "block" }}>
      {data.map((d, i) => {
        const barH = Math.max(2, (d.value / max) * height);
        const x = i * (barW + gap);
        const y = height - barH;
        return (
          <g key={i}>
            <rect x={x} y={y} width={barW} height={barH} fill={color} opacity={0.85} rx={1} />
            {data.length <= 30 && (
              <text x={x + barW / 2} y={height + 12} textAnchor="middle" fontSize={8} fill="#475569">
                {d.label}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

interface SkipTimelinePoint { tick: number; label: string; skip_rate: number; completed: number; skipped: number; }
interface SkipTimelineSeries { strategy: string; points: SkipTimelinePoint[]; }
interface SkipTimeline { window: number; step?: number; max_tick: number; strategies: SkipTimelineSeries[]; }

// Paper-ready line graph: skip rate per strategy, computed over each window of ticks.
function SkipRateTimelineChart({ data }: { data: SkipTimeline | null }) {
  const series = (data?.strategies ?? []).filter((s) => s.points.length > 0);
  if (series.length === 0) {
    return <div style={{ color: "#334155", fontSize: "12px" }}>No recommendations logged yet.</div>;
  }
  const W = 700, H = 330, padL = 74, padR = 18, padT = 14, padB = 72;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const ticks = Array.from(new Set(series.flatMap((s) => s.points.map((p) => p.tick)))).sort((a, b) => a - b);
  const tickMax = Math.max(...ticks, 1);
  const win = data?.window ?? 10;
  const x = (t: number) => padL + (tickMax === 0 ? plotW / 2 : (t / tickMax) * plotW);
  const y = (r: number) => padT + plotH * (1 - Math.max(0, Math.min(1, r)));
  const gridVals = [0, 0.2, 0.4, 0.6, 0.8, 1];

  // Thin the x-axis to ~8 evenly spaced, round-numbered labels — with a sliding
  // window there can be 40+ points and per-point labels mash into each other.
  const labelStride = Math.max(1, Math.ceil(ticks.length / 8));
  const labeledTicks = ticks.filter((_, i) => i % labelStride === 0);
  if (labeledTicks[labeledTicks.length - 1] !== ticks[ticks.length - 1]) {
    const last = ticks[ticks.length - 1];
    // Only append the final tick if it won't crowd the previous label.
    if (x(last) - x(labeledTicks[labeledTicks.length - 1]) > 34) labeledTicks.push(last);
  }

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
      {/* Y gridlines + labels (skip rate %) */}
      {gridVals.map((v) => (
        <g key={v}>
          <line x1={padL} y1={y(v)} x2={W - padR} y2={y(v)} stroke="rgba(255,255,255,0.08)" strokeWidth={1} />
          <text x={padL - 10} y={y(v) + 4} textAnchor="end" fontSize={12} fill="#94a3b8">{(v * 100).toFixed(0)}%</text>
        </g>
      ))}
      {/* X ticks: thinned, larger, with tick marks */}
      {labeledTicks.map((t) => (
        <g key={t}>
          <line x1={x(t)} y1={padT + plotH} x2={x(t)} y2={padT + plotH + 5} stroke="rgba(255,255,255,0.25)" strokeWidth={1} />
          <text x={x(t)} y={padT + plotH + 21} textAnchor="middle" fontSize={12} fill="#94a3b8">{t}</text>
        </g>
      ))}
      <text x={padL + plotW / 2} y={H - 26} textAnchor="middle" fontSize={12} fill="#94a3b8">
        window start tick{win ? `  (each point = skip rate over ticks t … t+${win - 1})` : ""}
      </text>
      <text x={16} y={padT + plotH / 2} textAnchor="middle" fontSize={12} fill="#94a3b8"
        transform={`rotate(-90 16 ${padT + plotH / 2})`}>skip rate (%)</text>

      {/* One line per strategy */}
      {series.map((s) => {
        const color = STRATEGY_COLORS[s.strategy] ?? "#6366f1";
        const pts = [...s.points].sort((a, b) => a.tick - b.tick);
        const path = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.tick)} ${y(p.skip_rate)}`).join(" ");
        return (
          <g key={s.strategy}>
            <path d={path} fill="none" stroke={color} strokeWidth={2.5} strokeLinejoin="round" />
            {pts.map((p) => (
              <circle key={p.tick} cx={x(p.tick)} cy={y(p.skip_rate)} r={3.5} fill={color}>
                <title>{`${s.strategy} · ticks ${p.label}: ${(p.skip_rate * 100).toFixed(1)}% skip (${p.skipped}/${p.completed})`}</title>
              </circle>
            ))}
          </g>
        );
      })}
      {/* Axis line */}
      <line x1={padL} y1={padT + plotH} x2={W - padR} y2={padT + plotH} stroke="rgba(255,255,255,0.18)" strokeWidth={1} />

      {/* Legend — bottom row, clear of the axis title */}
      {series.map((s, i) => {
        const lx = padL + i * 150;
        const color = STRATEGY_COLORS[s.strategy] ?? "#6366f1";
        return (
          <g key={s.strategy} transform={`translate(${lx}, ${H - 6})`}>
            <line x1={0} y1={-4} x2={20} y2={-4} stroke={color} strokeWidth={3} />
            <circle cx={10} cy={-4} r={3.5} fill={color} />
            <text x={27} y={0} fontSize={12} fontWeight={600} fill="#cbd5e1">{s.strategy}</text>
          </g>
        );
      })}
    </svg>
  );
}

interface UMLeaf { label: string; n: number; engage_prob: number; entropy: number; examples: string[]; }
interface UMNode { kind: "split" | "leaf"; n: number; engage_prob: number; entropy: number; label?: string; feature?: number; threshold?: number; children: UMNode[]; }
interface UMAgent { id: string; name: string; avatar_color: string; domain: string; n_events: number; accuracy: number | null; train_accuracy: number | null; updated_tick: number; leaves: UMLeaf[]; tree?: UMNode | null; }
interface UserModelTree { agents: UMAgent[]; }

// Draw the learned decision tree: internal nodes are PCA splits, leaves are the
// data-discovered topics (LLM-named). Colour = engage probability.
function DecisionTreeSVG({ root }: { root: UMNode }) {
  const NW = 148, NH = 46, HGAP = 18, VGAP = 58;

  // First pass: lay out leaves left-to-right, parents centred over their children.
  let cursor = 0;
  const place = (n: UMNode, depth: number): { x: number; y: number; node: UMNode; kids: number[] } => {
    const y = depth * VGAP;
    if (n.kind === "leaf" || n.children.length === 0) {
      const x = cursor * (NW + HGAP);
      cursor += 1;
      const idx = nodes.length;
      nodes.push({ x, y, node: n, kids: [] });
      return nodes[idx];
    }
    const kids = n.children.map((c) => place(c, depth + 1));
    const x = (kids[0].x + kids[kids.length - 1].x) / 2;
    const idx = nodes.length;
    nodes.push({ x, y, node: n, kids: kids.map((k) => nodes.indexOf(k)) });
    return nodes[idx];
  };
  const nodes: { x: number; y: number; node: UMNode; kids: number[] }[] = [];
  place(root, 0);

  const maxX = Math.max(...nodes.map((n) => n.x)) + NW;
  const maxY = Math.max(...nodes.map((n) => n.y)) + NH;
  const W = maxX + 16, H = maxY + 26;

  const fill = (p: number) => (p >= 0.75 ? "#10b981" : p >= 0.5 ? "#3f9e7c" : p >= 0.25 ? "#a16207" : "#ef4444");

  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: "block", minWidth: "100%" }}>
        {/* edges */}
        {nodes.map((p, i) =>
          p.kids.map((ki, j) => {
            const c = nodes[ki];
            const x1 = p.x + NW / 2, y1 = p.y + NH, x2 = c.x + NW / 2, y2 = c.y;
            const my = (y1 + y2) / 2;
            return (
              <g key={`${i}-${ki}`}>
                <path d={`M ${x1} ${y1} L ${x1} ${my} L ${x2} ${my} L ${x2} ${y2}`}
                  fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth={1.5} />
                <text x={(x1 + x2) / 2} y={my - 4} textAnchor="middle" fontSize={9} fill="#475569">
                  {j === 0 ? "yes" : "no"}
                </text>
              </g>
            );
          })
        )}
        {/* nodes */}
        {nodes.map((p, i) => {
          const n = p.node;
          const isLeaf = n.kind === "leaf" || n.children.length === 0;
          const c = fill(n.engage_prob);
          return (
            <g key={i}>
              <rect x={p.x} y={p.y} width={NW} height={NH} rx={6}
                fill={isLeaf ? c : "#0f172a"} fillOpacity={isLeaf ? 0.18 : 1}
                stroke={isLeaf ? c : "rgba(255,255,255,0.16)"} strokeWidth={isLeaf ? 1.5 : 1} />
              <title>
                {isLeaf
                  ? `${n.label || "unlabeled"} — ${(n.engage_prob * 100).toFixed(0)}% engage, ${n.n} samples, entropy ${n.entropy}`
                  : `split on PCA component ${n.feature} at ${n.threshold} — ${n.n} samples, entropy ${n.entropy}`}
              </title>
              {isLeaf ? (
                <>
                  <text x={p.x + NW / 2} y={p.y + 18} textAnchor="middle" fontSize={10.5} fontWeight={600} fill="#e2e8f0">
                    {(n.label || "unlabeled").length > 21 ? (n.label || "").slice(0, 20) + "…" : n.label || "unlabeled"}
                  </text>
                  <text x={p.x + NW / 2} y={p.y + 34} textAnchor="middle" fontSize={10} fill={c} fontFamily="monospace">
                    {(n.engage_prob * 100).toFixed(0)}% engage · n{n.n}
                  </text>
                </>
              ) : (
                <>
                  <text x={p.x + NW / 2} y={p.y + 18} textAnchor="middle" fontSize={10.5} fontFamily="monospace" fill="#cbd5e1">
                    PC{n.feature} ≤ {n.threshold}
                  </text>
                  <text x={p.x + NW / 2} y={p.y + 34} textAnchor="middle" fontSize={9.5} fill="#64748b">
                    n{n.n} · H {n.entropy.toFixed(2)}
                  </text>
                </>
              )}
            </g>
          );
        })}
      </svg>
      <div style={{ display: "flex", gap: "14px", alignItems: "center", marginTop: "8px", fontSize: "10px", color: "#475569", flexWrap: "wrap" }}>
        <span>leaf colour = engage rate:</span>
        {[["#ef4444", "always skips"], ["#a16207", "mixed"], ["#10b981", "always engages"]].map(([c, l]) => (
          <span key={l} style={{ display: "flex", alignItems: "center", gap: "5px" }}>
            <span style={{ width: "10px", height: "10px", borderRadius: "2px", background: c as string, opacity: 0.5, border: `1px solid ${c}` }} />
            {l}
          </span>
        ))}
        <span style={{ marginLeft: "auto" }}>PC = PCA component of the post embedding · H = label entropy (0 = pure)</span>
      </div>
    </div>
  );
}
interface InfAgent { agent_id: string; agent_name: string; avatar_color: string; accuracy: number | null; train_accuracy: number | null; n_events: number; }
interface InferenceAccuracy { per_agent: InfAgent[]; avg_accuracy: number | null; agents_modeled: number; has_ground_truth: boolean; }

function accColor(a: number | null): string {
  if (a === null) return "#475569";
  return a >= 0.85 ? "#10b981" : a >= 0.65 ? "#f59e0b" : "#ef4444";
}

// The learned decision-tree user model: per-agent inference accuracy + discovered leaves.
function UserModelPanel({ inference, tree }: { inference: InferenceAccuracy | null; tree: UserModelTree | null }) {
  const [openAgent, setOpenAgent] = useState<string>("");
  if (!inference || inference.agents_modeled === 0) {
    return (
      <div style={{ color: "#475569", fontSize: "12px" }}>
        No decision-tree user models yet. Run a <b>synthetic</b> session (Simulation → Agent decision mode)
        so the clickstream trains the trees, or POST train_user_model.
      </div>
    );
  }
  const leavesById: Record<string, UMLeaf[]> = {};
  const treeById: Record<string, UMNode | null | undefined> = {};
  for (const a of tree?.agents ?? []) { leavesById[a.id] = a.leaves ?? []; treeById[a.id] = a.tree; }
  return (
    <div>
      <div style={{ display: "flex", alignItems: "baseline", gap: "16px", marginBottom: "12px", flexWrap: "wrap" }}>
        {inference.has_ground_truth && (
          <div>
            <span style={{ fontSize: "26px", fontWeight: 700, color: accColor(inference.avg_accuracy), lineHeight: 1 }}>
              {inference.avg_accuracy !== null ? `${(inference.avg_accuracy * 100).toFixed(1)}%` : "—"}
            </span>
            <span style={{ fontSize: "11px", color: "#64748b", marginLeft: "8px" }}>
              mean inference accuracy vs ground truth
            </span>
          </div>
        )}
        <span style={{ fontSize: "11px", color: "#475569", marginLeft: "auto" }}>
          {inference.agents_modeled} agent model{inference.agents_modeled === 1 ? "" : "s"} · tree predicts skip from post embedding
        </span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
        {inference.per_agent.map((a) => {
          const leaves = leavesById[a.agent_id] ?? [];
          const open = openAgent === a.agent_id;
          return (
            <div key={a.agent_id} style={{ background: "#0f172a", borderRadius: "6px", padding: "8px 12px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "10px", fontSize: "12px", cursor: leaves.length ? "pointer" : "default" }}
                onClick={() => leaves.length && setOpenAgent(open ? "" : a.agent_id)}>
                <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: a.avatar_color, flexShrink: 0 }} />
                <span style={{ width: "150px", color: "#cbd5e1", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{a.agent_name}</span>
                <span style={{ fontFamily: "monospace", color: accColor(a.accuracy), width: "90px" }}>
                  {a.accuracy !== null ? `acc ${(a.accuracy * 100).toFixed(0)}%` : "acc —"}
                </span>
                <span style={{ fontFamily: "monospace", color: "#64748b", fontSize: "10px", width: "70px" }}>
                  fit {a.train_accuracy !== null ? `${(a.train_accuracy * 100).toFixed(0)}%` : "—"}
                </span>
                <span style={{ fontSize: "10px", color: "#475569" }}>{a.n_events} events</span>
                {leaves.length > 0 && <span style={{ marginLeft: "auto", fontSize: "10px", color: "#475569" }}>{open ? "▾" : "▸"} {leaves.length} leaves</span>}
              </div>
              {open && (
                <div style={{ marginTop: "10px" }}>
                  {treeById[a.agent_id] ? (
                    <DecisionTreeSVG root={treeById[a.agent_id] as UMNode} />
                  ) : (
                    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                      <div style={{ fontSize: "10px", color: "#f59e0b", marginBottom: "4px" }}>
                        No tree structure stored for this agent yet — click <b>Retrain user model</b> to generate it.
                      </div>
                      {leaves.map((lf, i) => (
                        <div key={i} title={lf.examples.join("  ·  ")} style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "11px", padding: "3px 6px", background: "rgba(255,255,255,0.02)", borderRadius: "4px" }}>
                          <span style={{ flex: 1, color: "#cbd5e1", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{lf.label || "(unlabeled)"}</span>
                          <span style={{ fontFamily: "monospace", color: lf.engage_prob >= 0.5 ? "#10b981" : "#ef4444" }}>
                            {(lf.engage_prob * 100).toFixed(0)}% engage
                          </span>
                          <span style={{ fontFamily: "monospace", color: "#475569", width: "34px", textAlign: "right" }}>n{lf.n}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div style={{ ...card, flex: 1, minWidth: 0 }}>
      <div style={{ fontSize: "11px", color: "#475569", textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: "6px" }}>{label}</div>
      <div style={{ fontSize: "28px", fontWeight: 700, color: "#00d4ff", lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: "11px", color: "#334155", marginTop: "4px" }}>{sub}</div>}
    </div>
  );
}

export default function AnalyticsPage() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState("");
  const [stats, setStats] = useState<StrategyStats[]>([]);
  const [skipTimeline, setSkipTimeline] = useState<SkipTimeline | null>(null);
  const [inference, setInference] = useState<InferenceAccuracy | null>(null);
  const [userTree, setUserTree] = useState<UserModelTree | null>(null);
  const [retraining, setRetraining] = useState(false);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [growth, setGrowth] = useState<GrowthPoint[]>([]);
  const [agentStats, setAgentStats] = useState<AgentStat[]>([]);
  const [activity, setActivity] = useState<ActivityPoint[]>([]);

  useEffect(() => {
    getSessions().then((s: Session[]) => {
      setSessions(s);
      if (s.length > 0) setActiveId(s[0].id);
    }).catch(() => {});
  }, []);

  const refresh = useCallback(() => {
    if (!activeId) return;
    getRecommendationStats(activeId).then(setStats).catch(() => {});
    getSkipRateTimeline(activeId, 10, 1).then(setSkipTimeline).catch(() => setSkipTimeline(null));
    getInferenceAccuracy(activeId).then(setInference).catch(() => setInference(null));
    getUserModelTree(activeId).then(setUserTree).catch(() => setUserTree(null));
    getSessionProfiles(activeId).then(setProfiles).catch(() => {});
    getTreeGrowth(activeId).then(setGrowth).catch(() => {});
    getAgentStats(activeId).then(setAgentStats).catch(() => {});
    getActivityPerTick(activeId).then(setActivity).catch(() => {});
  }, [activeId]);

  useEffect(() => {
    refresh();
    const iv = setInterval(refresh, 5000);
    return () => clearInterval(iv);
  }, [refresh]);

  // Derived summary stats
  const activeSession = sessions.find((s) => s.id === activeId);
  const totalPosts = growth.length > 0
    ? Object.values(
        growth.reduce((acc, g) => { acc[g.ptype] = g.count; return acc; }, {} as Record<string, number>)
      ).reduce((a, b) => a + b, 0)
    : 0;
  const totalEvents = activity.reduce((a, b) => a + b.count, 0);
  const topContributor = agentStats.length > 0 ? agentStats[0] : null;

  // Latest count per ptype
  const ptypeLatest: Record<string, number> = {};
  for (const g of growth) {
    ptypeLatest[g.ptype] = g.count;
  }
  const ptypes = Object.keys(ptypeLatest);
  const maxPtypeCount = Math.max(...Object.values(ptypeLatest), 1);
  const maxAgentTotal = agentStats.length > 0 ? agentStats[0].total : 1;

  return (
    <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "28px 24px", color: "#e2e8f0" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "20px" }}>
        <h1 style={{ fontSize: "22px", fontWeight: 700, color: "#00d4ff", margin: 0 }}>Analytics</h1>
        <select
          value={activeId}
          onChange={(e) => setActiveId(e.target.value)}
          style={{ background: "#0f172a", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "6px", padding: "7px 12px", color: "#e2e8f0", fontSize: "13px" }}
        >
          {sessions.map((s) => (
            <option key={s.id} value={s.id}>
              {s.id.slice(0, 8)} — {s.status} — t{s.tick_count}
            </option>
          ))}
        </select>
      </div>

      {/* Summary row */}
      <div style={{ display: "flex", gap: "12px", marginBottom: "20px", flexWrap: "wrap" }}>
        <StatCard label="Total Posts" value={totalPosts} />
        <StatCard label="Ticks Run" value={activeSession?.tick_count ?? 0} />
        <StatCard label="Total Events" value={totalEvents} />
        <StatCard
          label="Top Contributor"
          value={topContributor?.agent_name ?? "—"}
          sub={topContributor ? `${topContributor.total} posts` : undefined}
        />
      </div>

      {/* Skip rate by strategy — key paper figure */}
      <div style={{ ...card, marginBottom: "16px" }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: "10px", flexWrap: "wrap", gap: "8px" }}>
          <h2 style={{ fontSize: "15px", fontWeight: 600, margin: 0, color: "#e2e8f0" }}>
            Skip Rate over Time
          </h2>
          <span style={{ fontSize: "11px", color: "#475569" }}>
            sliding {skipTimeline?.window ?? 10}-tick window (step {skipTimeline?.step ?? 1}) — lower is better
          </span>
        </div>
        <SkipRateTimelineChart data={skipTimeline} />
      </div>

      {/* Decision-tree user model + inference accuracy */}
      <div style={{ ...card, marginBottom: "16px" }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: "10px", flexWrap: "wrap", gap: "8px" }}>
          <h2 style={{ fontSize: "15px", fontWeight: 600, margin: 0, color: "#e2e8f0" }}>
            Decision-tree user model
          </h2>
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <span style={{ fontSize: "11px", color: "#475569" }}>
              learned from the clickstream · click an agent to see its tree
            </span>
            <button
              onClick={async () => {
                if (!activeId || retraining) return;
                setRetraining(true);
                try { await trainUserModel(activeId); refresh(); }
                catch { /* surfaced by the panel's empty state */ }
                finally { setRetraining(false); }
              }}
              disabled={!activeId || retraining}
              style={{ background: retraining ? "#1e293b" : "#0f172a", color: retraining ? "#64748b" : "#cbd5e1", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "6px", padding: "6px 12px", fontSize: "11.5px", fontWeight: 600, cursor: "pointer" }}
            >
              {retraining ? "Retraining…" : "Retrain user model"}
            </button>
          </div>
        </div>
        <UserModelPanel inference={inference} tree={userTree} />
      </div>

      {/* Row 2: activity + strategy */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: "16px", marginBottom: "16px" }}>

        {/* Activity timeline */}
        <div style={card}>
          <h2 style={{ fontSize: "14px", fontWeight: 600, marginBottom: "12px", color: "#e2e8f0" }}>
            Activity per Tick
          </h2>
          {activity.length === 0 ? (
            <div style={{ color: "#334155", fontSize: "12px" }}>No activity yet.</div>
          ) : (
            <SvgBarChart
              data={activity.map((a) => ({ label: a.tick, value: a.count }))}
              color="#00d4ff"
              height={80}
            />
          )}
        </div>

        {/* Strategy metrics */}
        <div style={card}>
          <h2 style={{ fontSize: "14px", fontWeight: 600, marginBottom: "14px", color: "#e2e8f0" }}>
            Recommendation Strategies
          </h2>
          {stats.length === 0 ? (
            <div style={{ color: "#334155", fontSize: "12px" }}>No data yet.</div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
              {stats.map((s) => (
                <div key={s.strategy}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "5px", alignItems: "center" }}>
                    <span style={{ color: STRATEGY_COLORS[s.strategy] ?? "#94a3b8", fontWeight: 700, fontSize: "12px" }}>
                      {s.strategy}
                    </span>
                    <span style={{ fontSize: "10px", color: "#475569" }}>n={s.total_recommendations}</span>
                  </div>
                  <div style={{ display: "flex", gap: "6px", fontSize: "10px", marginBottom: "6px" }}>
                    <span style={{ background: "rgba(16,185,129,0.15)", color: "#10b981", padding: "2px 6px", borderRadius: "3px" }}>
                      {s.engaged} engaged
                    </span>
                    <span style={{ background: "rgba(239,68,68,0.12)", color: "#f87171", padding: "2px 6px", borderRadius: "3px" }}>
                      {s.skipped} skipped
                    </span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "3px" }}>
                    <span style={{ fontSize: "10px", color: "#475569", width: "54px" }}>skip rate</span>
                    <div style={{ flex: 1, background: "rgba(255,255,255,0.05)", borderRadius: "3px", height: "8px" }}>
                      <div style={{ height: "100%", width: `${s.skip_rate * 100}%`, background: s.skip_rate > 0.5 ? "#ef4444" : "#f97316", borderRadius: "3px" }} />
                    </div>
                    <span style={{ fontSize: "11px", color: "#94a3b8", width: "34px", textAlign: "right" }}>{(s.skip_rate * 100).toFixed(0)}%</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <span style={{ fontSize: "10px", color: "#475569", width: "54px" }}>accuracy</span>
                    <div style={{ flex: 1, background: "rgba(255,255,255,0.05)", borderRadius: "3px", height: "8px" }}>
                      <div style={{ height: "100%", width: `${s.accuracy * 100}%`, background: STRATEGY_COLORS[s.strategy] ?? "#6366f1", borderRadius: "3px" }} />
                    </div>
                    <span style={{ fontSize: "11px", color: "#94a3b8", width: "34px", textAlign: "right" }}>{(s.accuracy * 100).toFixed(0)}%</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Row 3: post types + agent contributions */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginBottom: "16px" }}>

        {/* Post type distribution */}
        <div style={card}>
          <h2 style={{ fontSize: "14px", fontWeight: 600, marginBottom: "14px", color: "#e2e8f0" }}>
            Post Type Distribution
          </h2>
          {ptypes.length === 0 ? (
            <div style={{ color: "#334155", fontSize: "12px" }}>No posts yet.</div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: "7px" }}>
              {ptypes
                .sort((a, b) => ptypeLatest[b] - ptypeLatest[a])
                .map((ptype) => {
                  const count = ptypeLatest[ptype];
                  const pct = (count / maxPtypeCount) * 100;
                  return (
                    <div key={ptype} style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "12px" }}>
                      <span style={{ width: "90px", color: PTYPE_COLORS[ptype] ?? "#94a3b8", fontWeight: 600, flexShrink: 0 }}>{ptype}</span>
                      <div style={{ flex: 1, background: "rgba(255,255,255,0.05)", borderRadius: "3px", height: "9px" }}>
                        <div style={{ height: "100%", width: `${pct}%`, background: PTYPE_COLORS[ptype] ?? "#6366f1", borderRadius: "3px", transition: "width 0.5s" }} />
                      </div>
                      <span style={{ color: "#64748b", width: "22px", textAlign: "right", flexShrink: 0 }}>{count}</span>
                    </div>
                  );
                })}
            </div>
          )}
        </div>

        {/* Agent contributions */}
        <div style={card}>
          <h2 style={{ fontSize: "14px", fontWeight: 600, marginBottom: "14px", color: "#e2e8f0" }}>
            Agent Contributions
          </h2>
          {agentStats.length === 0 ? (
            <div style={{ color: "#334155", fontSize: "12px" }}>No posts yet.</div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: "7px" }}>
              {agentStats.map((a) => (
                <div key={a.agent_id} style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "12px" }}>
                  <div style={{ width: "7px", height: "7px", borderRadius: "50%", background: a.avatar_color, flexShrink: 0 }} />
                  <span style={{ width: "110px", color: "#cbd5e1", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flexShrink: 0, fontSize: "11px" }}>
                    {a.agent_name}
                  </span>
                  <div style={{ flex: 1, background: "rgba(255,255,255,0.05)", borderRadius: "3px", height: "9px" }}>
                    <div style={{ height: "100%", width: `${(a.total / maxAgentTotal) * 100}%`, background: a.avatar_color, borderRadius: "3px", transition: "width 0.5s" }} />
                  </div>
                  <span style={{ color: "#64748b", width: "22px", textAlign: "right", flexShrink: 0 }}>{a.total}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Agent Profiles */}
      <div style={card}>
        <h2 style={{ fontSize: "14px", fontWeight: 600, marginBottom: "14px", color: "#e2e8f0" }}>
          Agent Profiles
          <span style={{ fontSize: "10px", color: "#334155", fontWeight: 400, marginLeft: "8px" }}>built by multi_agent strategy</span>
        </h2>
        {profiles.length === 0 ? (
          <div style={{ color: "#475569", fontSize: "12px" }}>No profiles yet. Multi-agent strategy builds these over time.</div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: "10px" }}>
            {profiles.map((p) => (
              <div key={p.agent_id} style={{ background: "#0f172a", borderRadius: "6px", padding: "12px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
                  <span style={{ color: "#00d4ff", fontWeight: 600, fontSize: "12px" }}>
                    {p.agent_name ?? p.agent_id.slice(0, 8)}
                  </span>
                  <span style={{ color: "#334155", fontSize: "10px" }}>t{p.last_updated_tick}</span>
                </div>
                <p style={{ color: "#64748b", fontSize: "11px", lineHeight: 1.55, margin: 0 }}>
                  {p.profile_text || "—"}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
