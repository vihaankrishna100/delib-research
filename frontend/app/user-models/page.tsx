"use client";
import { useState, useEffect, useCallback, useMemo } from "react";
import {
  getSessions,
  getUserModels,
  getPriors,
  getResearchMetrics,
  rebuildTopicTree,
  computePriors,
  scoreQuality,
  generatePersonaTopics,
  getPersonaTopics,
  getPersonaUserModels,
  refinePersonaTopics,
  scorePersonaActions,
  splitPersonaTopic,
  createUserFromDescription,
  getAgentReports,
} from "@/lib/api";

interface Session { id: string; status: string; tick_count: number; }
interface AgentMeta { id: string; name: string; domain: string; avatar_color: string; role: string; }
interface TopicNode { id: string; parent_id: string | null; label: string; level: number; is_leaf: boolean; member_post_ids: string[]; }
interface ObservedCell { agent_id: string; topic_id: string; score: number; posts_authored: number; engaged: number; skipped: number; }
interface PriorCellT { agent_id: string; topic_id: string; score: number; rationale?: string | null; }
interface UserModelMatrix { agents: AgentMeta[]; topics: TopicNode[]; cells: ObservedCell[]; }
interface PriorMatrix { agents: AgentMeta[]; topics: TopicNode[]; cells: PriorCellT[]; }
interface AgentResearch { agent_id: string; agent_name: string; avatar_color: string; persona_fidelity: number; skip_rate: number; avg_post_quality: number | null; posts_authored: number; engaged: number; skipped: number; }
interface TopicSurprise { topic_id: string; label: string; mean_abs_drift: number; }
interface ResearchMetrics { per_agent: AgentResearch[]; per_topic_surprise: TopicSurprise[]; avg_persona_fidelity: number; avg_skip_rate: number; avg_post_quality: number | null; quality_scored_posts: number; total_posts: number; }
interface PersonaHeader { id: string; agent_id: string; label: string; rank: number; rationale?: string | null; }
interface PersonaTopics { agents: AgentMeta[]; headers: PersonaHeader[]; }
interface AgentReport { agent_id: string; name: string; text: string; }
interface UsageTopic { id: string; parent_id?: string | null; label: string; rationale?: string | null; rank: number; confidence: number; action_scores?: Record<string, number>; observed_actions?: Record<string, number>; posts: number; share: number; used: boolean; examples: string[]; subtopics?: UsageTopic[]; }
interface UsageAgent { id: string; name: string; domain: string; avatar_color: string; role: string; total_posts: number; entropy: number; action_categories?: string[]; topics: UsageTopic[]; }
interface PersonaUsage { agents: UsageAgent[]; }

type View = "observed" | "prior" | "drift";

const PANEL: React.CSSProperties = {
  background: "rgba(255,255,255,0.03)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: "10px",
  padding: "16px",
};

function heatColor(score: number): string {
  const t = Math.max(0, Math.min(1, score));
  const r = Math.round(6 + (0 - 6) * t);
  const g = Math.round(13 + (212 - 13) * t);
  const b = Math.round(26 + (255 - 26) * t);
  return `rgb(${r},${g},${b})`;
}

function driftColor(diff: number): string {
  // diff in [-1, 1]: red (under-engaged) → neutral → blue (over-engaged)
  const t = Math.max(-1, Math.min(1, diff));
  if (t >= 0) {
    const a = t; // 0..1
    const r = Math.round(15 * (1 - a));
    const g = Math.round(20 + (180 - 20) * a);
    const b = Math.round(30 + (255 - 30) * a);
    return `rgb(${r},${g},${b})`;
  } else {
    const a = -t;
    const r = Math.round(40 + (220 - 40) * a);
    const g = Math.round(15 * (1 - a));
    const b = Math.round(30 * (1 - a));
    return `rgb(${r},${g},${b})`;
  }
}

const HEADER_BTN: React.CSSProperties = {
  border: "none", borderRadius: "6px", padding: "7px 12px",
  fontSize: "12px", fontWeight: 600, cursor: "pointer",
};

function ViewToggleButton({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        ...HEADER_BTN,
        background: active ? "#00d4ff" : "transparent",
        color: active ? "#060d1a" : "#94a3b8",
        border: active ? "none" : "1px solid rgba(255,255,255,0.12)",
      }}
    >
      {label}
    </button>
  );
}

// The deliberation "move" types we predict/observe per topic header — i.e. what kind
// of contribution the agent would add in that header. Colors mirror the map's ptypes.
const ACTION_META: { key: string; label: string; color: string }[] = [
  { key: "goal", label: "goal", color: "#10b981" },
  { key: "barrier", label: "barrier", color: "#ef4444" },
  { key: "cause", label: "cause", color: "#f97316" },
  { key: "solution", label: "solution", color: "#00d4ff" },
  { key: "abstraction", label: "abstraction", color: "#8b5cf6" },
  { key: "analogy", label: "analogy", color: "#6366f1" },
  { key: "inspiration", label: "inspiration", color: "#f59e0b" },
  { key: "question", label: "question", color: "#ec4899" },
  { key: "answer", label: "answer", color: "#38bdf8" },
  { key: "pro", label: "pro", color: "#22c55e" },
  { key: "con", label: "con", color: "#dc2626" },
  { key: "part", label: "part", color: "#14b8a6" },
  { key: "failure_mode", label: "failure", color: "#f43f5e" },
];

// Compact per-action score strip: predicted score bar + observed count for each action.
function ActionScoreBars({ scores, observed }: { scores?: Record<string, number>; observed?: Record<string, number> }) {
  if (!scores || Object.keys(scores).length === 0) return null;
  const hasObserved = !!observed && ACTION_META.some((m) => (observed[m.key] ?? 0) > 0) || (observed?.skip ?? 0) > 0;
  return (
    <div style={{ display: "flex", gap: "12px", flexWrap: "wrap", padding: "2px 0 8px 26px" }}>
      {ACTION_META.map((m) => {
        const v = scores[m.key] ?? 0;
        const obs = observed?.[m.key] ?? 0;
        return (
          <div key={m.key} style={{ width: "70px" }} title={`predicted ${v.toFixed(2)}${hasObserved ? ` · observed ${obs}` : ""}`}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "9px", marginBottom: "2px" }}>
              <span style={{ color: m.color, fontWeight: 600 }}>{m.label}</span>
              <span style={{ fontFamily: "monospace", color: "#94a3b8" }}>{v.toFixed(2)}</span>
            </div>
            <div style={{ background: "rgba(255,255,255,0.06)", height: "5px", borderRadius: "3px" }}>
              <div style={{ width: `${Math.max(0, Math.min(1, v)) * 100}%`, height: "100%", background: m.color, borderRadius: "3px" }} />
            </div>
            {hasObserved && <div style={{ fontSize: "8px", color: "#475569", marginTop: "2px" }}>obs {obs}</div>}
          </div>
        );
      })}
      {hasObserved && (observed?.skip ?? 0) > 0 && (
        <div style={{ width: "50px" }} title={`observed skips ${observed?.skip}`}>
          <div style={{ fontSize: "9px", color: "#ef4444", fontWeight: 600, marginBottom: "2px" }}>skip</div>
          <div style={{ fontSize: "8px", color: "#475569", marginTop: "9px" }}>obs {observed?.skip}</div>
        </div>
      )}
    </div>
  );
}

// One header row + its per-action scores; recurses into sub-headers when expanded.
function TopicTreeRow({ topic, depth, expanded, onToggle, onSplit, splittingId }: {
  topic: UsageTopic; depth: number; expanded: Set<string>;
  onToggle: (id: string) => void; onSplit: (id: string) => void; splittingId: string;
}) {
  const hasChildren = !!topic.subtopics && topic.subtopics.length > 0;
  const canExpand = hasChildren || topic.examples.length > 0;
  const isOpen = expanded.has(topic.id);
  return (
    <div style={{ marginBottom: "2px" }}>
      <div title={topic.rationale ?? ""} style={{ display: "flex", alignItems: "center", gap: "8px", padding: "5px 6px", borderRadius: "6px", opacity: topic.used ? 1 : 0.55 }}>
        <span onClick={() => canExpand && onToggle(topic.id)} style={{ width: "12px", color: "#64748b", fontSize: "10px", cursor: canExpand ? "pointer" : "default" }}>
          {canExpand ? (isOpen ? "▾" : "▸") : "·"}
        </span>
        <span onClick={() => canExpand && onToggle(topic.id)} style={{ color: topic.used ? "#cbd5e1" : "#64748b", fontSize: depth === 0 ? "13px" : "12.5px", fontWeight: depth === 0 ? 600 : 500, cursor: canExpand ? "pointer" : "default" }}>
          {topic.label}
        </span>
        <span title="confidence the agent engages here (low = likely to skip)"
          style={{ fontSize: "9px", fontFamily: "monospace", color: topic.confidence >= 0.66 ? "#10b981" : topic.confidence >= 0.4 ? "#f59e0b" : "#ef4444" }}>
          c{topic.confidence.toFixed(2)}
        </span>
        <span style={{ fontSize: "10px", fontFamily: "monospace", color: topic.used ? "#00d4ff" : "#475569", background: topic.used ? "rgba(0,212,255,0.1)" : "transparent", padding: "1px 6px", borderRadius: "10px" }}>
          {topic.posts}
        </span>
        <button onClick={() => onSplit(topic.id)} disabled={splittingId === topic.id}
          title="Deepen this topic: generate 2-3 finer sub-headers beneath it (LLM)."
          style={{ marginLeft: "auto", fontSize: "10px", background: "transparent", border: "1px solid rgba(255,255,255,0.12)", color: splittingId === topic.id ? "#475569" : "#94a3b8", borderRadius: "4px", padding: "1px 7px", cursor: "pointer" }}>
          {splittingId === topic.id ? "deepening…" : "deepen"}
        </button>
      </div>
      <ActionScoreBars scores={topic.action_scores} observed={topic.observed_actions} />
      {isOpen && (
        <div style={{ borderLeft: "1px dashed rgba(255,255,255,0.1)", marginLeft: "11px", paddingLeft: "14px", marginBottom: "6px" }}>
          {hasChildren
            ? topic.subtopics!.map((s) => (
                <TopicTreeRow key={s.id} topic={s} depth={depth + 1} expanded={expanded} onToggle={onToggle} onSplit={onSplit} splittingId={splittingId} />
              ))
            : topic.examples.map((name, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: "7px", padding: "3px 0", fontSize: "12px", color: "#94a3b8" }}>
                  <span style={{ color: "#334155" }}>—</span>
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{name}</span>
                </div>
              ))}
        </div>
      )}
    </div>
  );
}

export default function UserModelsPage() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState("");
  const [observed, setObserved] = useState<UserModelMatrix | null>(null);
  const [prior, setPrior] = useState<PriorMatrix | null>(null);
  const [metrics, setMetrics] = useState<ResearchMetrics | null>(null);
  const [view, setView] = useState<View>("observed");
  const [rebuilding, setRebuilding] = useState(false);
  const [computingPriors, setComputingPriors] = useState(false);
  const [scoringQ, setScoringQ] = useState(false);
  const [error, setError] = useState<string>("");
  const [priorError, setPriorError] = useState<string>("");

  // Feature 1: persona topic headers + per-agent usage tables
  const [personaTopics, setPersonaTopics] = useState<PersonaTopics | null>(null);
  const [personaUsage, setPersonaUsage] = useState<PersonaUsage | null>(null);
  const [loadingUsage, setLoadingUsage] = useState(false);
  const [showUnused, setShowUnused] = useState(true);
  const [genTopics, setGenTopics] = useState(false);
  const [treeAgentId, setTreeAgentId] = useState<string>("");        // which agent's tree is shown
  const [expandedTopics, setExpandedTopics] = useState<Set<string>>(new Set());
  const [refining, setRefining] = useState(false);
  const [scoringActions, setScoringActions] = useState(false);
  const [splittingId, setSplittingId] = useState<string>("");
  const [newUserName, setNewUserName] = useState("");
  const [newUserDesc, setNewUserDesc] = useState("");
  const [addingUser, setAddingUser] = useState(false);
  const [addUserMsg, setAddUserMsg] = useState("");
  // Feature 2: per-agent action documents
  const [reports, setReports] = useState<AgentReport[]>([]);
  const [loadingReports, setLoadingReports] = useState(false);
  const [openReportId, setOpenReportId] = useState<string>("");

  useEffect(() => {
    getSessions().then((s: Session[]) => {
      setSessions(s);
      if (s.length > 0) setActiveId(s[0].id);
    }).catch(() => {});
  }, []);

  const refresh = useCallback(() => {
    if (!activeId) return;
    setError(""); setPriorError("");
    getUserModels(activeId)
      .then((m: UserModelMatrix) => { setObserved(m); })
      .catch((e: Error) => { setObserved(null); setError(e.message); });
    getPriors(activeId)
      .then((m: PriorMatrix) => setPrior(m))
      .catch((e: Error) => { setPrior(null); setPriorError(e.message); });
    getResearchMetrics(activeId)
      .then((m: ResearchMetrics) => setMetrics(m))
      .catch(() => setMetrics(null));
    getPersonaTopics(activeId)
      .then((p: PersonaTopics) => setPersonaTopics(p))
      .catch(() => setPersonaTopics(null));
    setLoadingUsage(true);
    getPersonaUserModels(activeId)
      .then((u: PersonaUsage) => setPersonaUsage(u))
      .catch(() => setPersonaUsage(null))
      .finally(() => setLoadingUsage(false));
    setReports([]); setOpenReportId("");
  }, [activeId]);

  useEffect(() => { refresh(); }, [refresh]);

  // Keep a valid agent selected for the per-agent tree view.
  useEffect(() => {
    if (!personaUsage || personaUsage.agents.length === 0) return;
    if (!personaUsage.agents.some((a) => a.id === treeAgentId)) {
      setTreeAgentId(personaUsage.agents[0].id);
    }
  }, [personaUsage, treeAgentId]);

  const toggleTopic = (id: string) =>
    setExpandedTopics((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  const reloadUsage = async () => {
    if (!activeId) return;
    setLoadingUsage(true);
    const [u, p] = await Promise.all([
      getPersonaUserModels(activeId).catch(() => null),
      getPersonaTopics(activeId).catch(() => null),
    ]);
    setPersonaUsage(u); setPersonaTopics(p); setLoadingUsage(false);
  };

  const handleRefine = async () => {
    if (!activeId) return;
    setRefining(true);
    try { await refinePersonaTopics(activeId); await reloadUsage(); }
    catch (e) { setError((e as Error).message); }
    finally { setRefining(false); }
  };

  const handleScoreActions = async () => {
    if (!activeId) return;
    setScoringActions(true);
    try { await scorePersonaActions(activeId); await reloadUsage(); }
    catch (e) { setError((e as Error).message); }
    finally { setScoringActions(false); }
  };

  const handleSplit = async (topicId: string) => {
    if (!activeId) return;
    setSplittingId(topicId);
    try { await splitPersonaTopic(activeId, topicId); await reloadUsage(); }
    catch (e) { setError((e as Error).message); }
    finally { setSplittingId(""); }
  };

  const handleAddUser = async () => {
    if (!activeId || !newUserDesc.trim()) return;
    setAddingUser(true); setAddUserMsg("");
    try {
      const r = await createUserFromDescription(activeId, newUserName, newUserDesc);
      setAddUserMsg(r.created
        ? `Created "${r.new_agent.name}" — seeded from ${r.similar_to} (similarity ${Number(r.similarity).toFixed(2)}), ${r.topics_copied} topics inherited.`
        : `Could not create: ${r.reason}`);
      setNewUserName(""); setNewUserDesc("");
      await reloadUsage();
    } catch (e) { setAddUserMsg((e as Error).message); }
    finally { setAddingUser(false); }
  };

  const handleGenTopics = async () => {
    if (!activeId) return;
    setGenTopics(true);
    try {
      await generatePersonaTopics(activeId);
      const p = await getPersonaTopics(activeId);
      setPersonaTopics(p);
      setLoadingUsage(true);
      const u = await getPersonaUserModels(activeId).catch(() => null);
      setPersonaUsage(u);
      setLoadingUsage(false);
    }
    catch (e) { setError((e as Error).message); }
    finally { setGenTopics(false); }
  };

  const handleLoadReports = async () => {
    if (!activeId) return;
    setLoadingReports(true);
    try {
      const r = await getAgentReports(activeId);
      setReports(r.reports ?? []);
      if (r.reports?.length) setOpenReportId(r.reports[0].agent_id);
    } catch (e) { setError((e as Error).message); }
    finally { setLoadingReports(false); }
  };

  const handleRebuild = async () => {
    if (!activeId) return;
    setRebuilding(true); setError("");
    try { await rebuildTopicTree(activeId); refresh(); }
    catch (e) { setError((e as Error).message); }
    finally { setRebuilding(false); }
  };

  const handleComputePriors = async () => {
    if (!activeId) return;
    setComputingPriors(true); setPriorError("");
    try { await computePriors(activeId); refresh(); }
    catch (e) { setPriorError((e as Error).message); }
    finally { setComputingPriors(false); }
  };

  const handleScoreQuality = async () => {
    if (!activeId) return;
    setScoringQ(true);
    try { await scoreQuality(activeId); refresh(); }
    catch (e) { setError((e as Error).message); }
    finally { setScoringQ(false); }
  };

  const matrix = observed; // matrix used for navigation structure

  const childrenOf = useCallback((pid: string | null) => {
    if (!matrix) return [];
    return matrix.topics
      .filter((t) => t.parent_id === pid)
      .sort((a, b) => a.label.localeCompare(b.label));
  }, [matrix]);

  const root = matrix?.topics.find((t) => t.parent_id === null) ?? null;

  // The cross-agent heatmap is intentionally locked to the TOP-LEVEL clustered topics
  // only — no drilling into member posts (that was the confusing "topics inside topics").
  const displayedTopics = childrenOf(root?.id ?? null).filter((t) => t.level === 1);

  const observedMap = useMemo(() => {
    const m: Record<string, ObservedCell> = {};
    if (observed) for (const c of observed.cells) m[`${c.agent_id}|${c.topic_id}`] = c;
    return m;
  }, [observed]);

  const priorMap = useMemo(() => {
    const m: Record<string, PriorCellT> = {};
    if (prior) for (const c of prior.cells) m[`${c.agent_id}|${c.topic_id}`] = c;
    return m;
  }, [prior]);

  const cellValue = (aid: string, tid: string): { display: number; bg: string; tooltip: string } | null => {
    if (view === "observed") {
      const c = observedMap[`${aid}|${tid}`];
      if (!c) return null;
      return {
        display: c.score, bg: heatColor(c.score),
        tooltip: `observed ${c.score.toFixed(3)} · authored ${c.posts_authored} · engaged ${c.engaged} · skipped ${c.skipped}`,
      };
    }
    if (view === "prior") {
      const p = priorMap[`${aid}|${tid}`];
      if (!p) return null;
      return {
        display: p.score, bg: heatColor(p.score),
        tooltip: `prior ${p.score.toFixed(3)}${p.rationale ? ` — ${p.rationale}` : ""}`,
      };
    }
    // drift = observed - prior
    const c = observedMap[`${aid}|${tid}`];
    const p = priorMap[`${aid}|${tid}`];
    if (!c || !p) return null;
    const diff = c.score - p.score;
    return {
      display: diff, bg: driftColor(diff),
      tooltip: `observed ${c.score.toFixed(2)} − prior ${p.score.toFixed(2)} = ${diff >= 0 ? "+" : ""}${diff.toFixed(2)}`,
    };
  };

  const priorReady = !!prior && !priorError;
  const driftReady = priorReady && !!observed;

  return (
    <div style={{ maxWidth: "1400px", margin: "0 auto", padding: "28px 24px", color: "#e2e8f0" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "20px", flexWrap: "wrap", gap: "12px" }}>
        <div>
          <h1 style={{ fontSize: "22px", fontWeight: 700, color: "#00d4ff", margin: 0 }}>User Models</h1>
          <div style={{ fontSize: "12px", color: "#64748b", marginTop: "4px" }}>
            Agent × topic interest. Compare the persona prior (LLM from persona alone) against observed activity (which folds in the prelude at low weight) to evaluate the recommender.
          </div>
        </div>
        <div style={{ display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" }}>
          <select
            value={activeId}
            onChange={(e) => setActiveId(e.target.value)}
            style={{ background: "#0f172a", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "6px", padding: "7px 12px", color: "#e2e8f0", fontSize: "13px" }}
          >
            {sessions.map((s) => (
              <option key={s.id} value={s.id}>{s.id.slice(0, 8)} — {s.status} — t{s.tick_count}</option>
            ))}
          </select>
          <button onClick={handleRebuild} disabled={!activeId || rebuilding}
            style={{ ...HEADER_BTN, background: rebuilding ? "#1e293b" : "#0f172a", color: rebuilding ? "#64748b" : "#cbd5e1", border: "1px solid rgba(255,255,255,0.12)" }}>
            {rebuilding ? "Rebuilding…" : "Rebuild topic tree"}
          </button>
          <button onClick={handleComputePriors} disabled={!activeId || computingPriors}
            title="Compute the persona prior — LLM scores each agent's persona against the topics (0..1). Build the topic tree first."
            style={{ ...HEADER_BTN, background: computingPriors ? "#1e293b" : "#0f172a", color: computingPriors ? "#64748b" : "#cbd5e1", border: "1px solid rgba(255,255,255,0.12)" }}>
            {computingPriors ? "Computing…" : "Compute priors"}
          </button>
          <button onClick={handleScoreQuality} disabled={!activeId || scoringQ}
            style={{ ...HEADER_BTN, background: scoringQ ? "#1e293b" : "#0f172a", color: scoringQ ? "#64748b" : "#cbd5e1", border: "1px solid rgba(255,255,255,0.12)" }}>
            {scoringQ ? "Scoring…" : "Score quality"}
          </button>
          <button onClick={handleGenTopics} disabled={!activeId || genTopics}
            title="Ask the LLM, in each agent's voice, for ~10 topics it expects to contribute to — seeds the persona table without a cold start."
            style={{ ...HEADER_BTN, background: genTopics ? "#1e293b" : "#0f172a", color: genTopics ? "#64748b" : "#cbd5e1", border: "1px solid rgba(255,255,255,0.12)" }}>
            {genTopics ? "Generating…" : "Persona topics"}
          </button>
          <button onClick={handleLoadReports} disabled={!activeId || loadingReports}
            title="Generate a per-agent document explaining each recommendation, how it was created, what the agent did, and why it skipped."
            style={{ ...HEADER_BTN, background: loadingReports ? "#1e293b" : "#0f172a", color: loadingReports ? "#64748b" : "#cbd5e1", border: "1px solid rgba(255,255,255,0.12)" }}>
            {loadingReports ? "Building…" : "Agent reports"}
          </button>
        </div>
      </div>

      {/* ── Per-agent topic trees — each agent owns their 10 topics ── */}
      <div style={{ ...PANEL, marginBottom: "18px" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: "12px", marginBottom: "10px", flexWrap: "wrap" }}>
          <h2 style={{ margin: 0, fontSize: "17px", color: "#00d4ff", fontWeight: 600 }}>Per-agent topic trees</h2>
          <span style={{ fontSize: "11px", color: "#64748b" }}>
            Each agent owns ~10 persona topics, each split into finer sub-headers. Expand to see sub-headers and posts. <b>c</b> = confidence; the bars under each header are predicted per-move scores (goal/barrier/cause/solution/analogy/…) — how likely the agent is to add each contribution type here — with observed counts below.
          </span>
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "10px" }}>
            <button onClick={handleScoreActions} disabled={!activeId || scoringActions}
              title="Predict a per-move score (goal/barrier/cause/solution/analogy/…) for every persona header from the agent's persona (LLM). Fills the bars under each header."
              style={{ ...HEADER_BTN, background: scoringActions ? "#1e293b" : "#0f172a", color: scoringActions ? "#64748b" : "#cbd5e1", border: "1px solid rgba(255,255,255,0.12)" }}>
              {scoringActions ? "Scoring…" : "Score actions"}
            </button>
            <button onClick={handleRefine} disabled={!activeId || refining}
              title="Re-score each topic's confidence from what the agent actually engaged with vs the reasons it skipped (semantic similarity), then reorder. Reduces the skip rate."
              style={{ ...HEADER_BTN, background: refining ? "#1e293b" : "#0f172a", color: refining ? "#64748b" : "#cbd5e1", border: "1px solid rgba(255,255,255,0.12)" }}>
              {refining ? "Refining…" : "Refine from skips"}
            </button>
            <label style={{ fontSize: "11px", color: "#94a3b8", display: "flex", alignItems: "center", gap: "6px", cursor: "pointer" }}>
              <input type="checkbox" checked={showUnused} onChange={(e) => setShowUnused(e.target.checked)} />
              Show unused topics
            </label>
          </div>
        </div>

        {loadingUsage && !personaUsage && (
          <div style={{ color: "#64748b", fontSize: "13px" }}>Computing per-agent topic trees…</div>
        )}
        {!loadingUsage && !personaUsage && (
          <div style={{ color: "#f59e0b", fontSize: "12px" }}>
            No persona topics yet — click <b>Persona topics</b> above to generate each agent&apos;s 10 topics.
          </div>
        )}

        {personaUsage && personaUsage.agents.length > 0 && (() => {
          const treeAgent = personaUsage.agents.find((a) => a.id === treeAgentId) ?? personaUsage.agents[0];
          const topics = treeAgent.topics
            .filter((t) => showUnused || t.used)
            .sort((x, y) => (y.posts - x.posts) || (x.rank - y.rank));
          return (
            <>
              {/* Agent switcher — "see between them" */}
              <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", marginBottom: "14px" }}>
                {personaUsage.agents.map((a) => {
                  const active = a.id === treeAgent.id;
                  return (
                    <button key={a.id} onClick={() => setTreeAgentId(a.id)}
                      style={{ ...HEADER_BTN, display: "flex", alignItems: "center", gap: "6px",
                        background: active ? "rgba(0,212,255,0.12)" : "transparent",
                        color: active ? "#e2e8f0" : "#94a3b8",
                        border: `1px solid ${active ? "#00d4ff" : "rgba(255,255,255,0.12)"}` }}>
                      <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: a.avatar_color }} />
                      {a.name}
                      <span style={{ fontSize: "10px", color: active ? "#00d4ff" : "#475569" }}>{a.total_posts}</span>
                    </button>
                  );
                })}
              </div>

              {/* The selected agent's tree */}
              <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: "10px", padding: "12px 16px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                  <span style={{ width: "11px", height: "11px", borderRadius: "50%", background: treeAgent.avatar_color }} />
                  <span style={{ color: "#e2e8f0", fontWeight: 700, fontSize: "15px" }}>{treeAgent.name}</span>
                  <span style={{ fontSize: "11px", color: "#475569" }}>{treeAgent.domain}</span>
                  <span style={{ marginLeft: "auto", fontSize: "11px", color: "#475569" }}>
                    {treeAgent.total_posts} post{treeAgent.total_posts === 1 ? "" : "s"} · {treeAgent.topics.filter((t) => t.used).length}/{treeAgent.topics.length} topics used
                    <span title="Entropy of the agent's post spread across topics (0 = focused on one topic, 1 = evenly spread). Like measuring diet variety."
                      style={{ marginLeft: "10px", color: "#a78bfa" }}>
                      variety {treeAgent.entropy.toFixed(2)}
                    </span>
                  </span>
                </div>

                {topics.length === 0 ? (
                  <div style={{ fontSize: "12px", color: "#475569", paddingLeft: "18px" }}>
                    No topics used yet. {treeAgent.total_posts === 0 ? "Run the simulation to populate the tree." : "Toggle “Show unused topics”."}
                  </div>
                ) : (
                  <div style={{ borderLeft: "1px solid rgba(255,255,255,0.08)", marginLeft: "5px", paddingLeft: "12px" }}>
                    {topics.map((t) => (
                      <TopicTreeRow
                        key={t.id}
                        topic={t}
                        depth={0}
                        expanded={expandedTopics}
                        onToggle={toggleTopic}
                        onSplit={handleSplit}
                        splittingId={splittingId}
                      />
                    ))}
                  </div>
                )}
              </div>
              <div style={{ marginTop: "10px", fontSize: "10px", color: "#475569" }}>
                Posts are matched to the agent&apos;s nearest sub-header by embedding similarity and rolled up to the parent. The number badge is how many of the agent&apos;s posts fall under each header; the bars are predicted per-move scores (goal/barrier/cause/solution/analogy/…) with observed counts beneath.
              </div>
            </>
          );
        })()}
      </div>

      {/* ── Add a new user from a short self-description ── */}
      {personaUsage && (
        <div style={{ ...PANEL, marginBottom: "18px" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: "12px", marginBottom: "10px", flexWrap: "wrap" }}>
            <h2 style={{ margin: 0, fontSize: "17px", color: "#00d4ff", fontWeight: 600 }}>Add a user from a description</h2>
            <span style={{ fontSize: "11px", color: "#64748b" }}>
              Describe yourself in a sentence or two. We find the most semantically similar existing persona and seed your topic table from theirs.
            </span>
          </div>
          <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", alignItems: "flex-start" }}>
            <input
              value={newUserName}
              onChange={(e) => setNewUserName(e.target.value)}
              placeholder="Name (optional)"
              style={{ width: "180px", background: "#0f172a", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "6px", padding: "8px 12px", color: "#e2e8f0", fontSize: "13px" }}
            />
            <textarea
              value={newUserDesc}
              onChange={(e) => setNewUserDesc(e.target.value)}
              placeholder="e.g. I'm a data scientist who cares about fairness in algorithms and community impact."
              rows={2}
              style={{ flex: 1, minWidth: "260px", background: "#0f172a", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "6px", padding: "8px 12px", color: "#e2e8f0", fontSize: "13px", resize: "vertical", boxSizing: "border-box" }}
            />
            <button onClick={handleAddUser} disabled={addingUser || !newUserDesc.trim()}
              style={{ ...HEADER_BTN, background: addingUser ? "#1e293b" : "#6366f1", color: "#fff", border: "none", padding: "9px 16px" }}>
              {addingUser ? "Adding…" : "Add user"}
            </button>
          </div>
          {addUserMsg && (
            <div style={{ marginTop: "10px", fontSize: "12px", color: addUserMsg.startsWith("Created") ? "#10b981" : "#f59e0b" }}>
              {addUserMsg}
            </div>
          )}
        </div>
      )}

      {/* ── Cross-agent topic heatmap (research view) ── */}
      <div style={{ fontSize: "13px", color: "#94a3b8", fontWeight: 600, marginBottom: "8px" }}>
        Cross-agent topic heatmap <span style={{ fontWeight: 400, color: "#475569", fontSize: "11px" }}>— research view: all agents × clustered topics (top level only)</span>
      </div>

      {/* View toggle */}
      <div style={{ display: "flex", gap: "8px", alignItems: "center", marginBottom: "14px", flexWrap: "wrap" }}>
        <ViewToggleButton label="Observed" active={view === "observed"} onClick={() => setView("observed")} />
        <ViewToggleButton label="Persona Prior" active={view === "prior"} onClick={() => priorReady ? setView("prior") : null} />
        <ViewToggleButton label="Drift (observed − prior)" active={view === "drift"} onClick={() => driftReady ? setView("drift") : null} />
        {!priorReady && view !== "observed" && (
          <span style={{ fontSize: "11px", color: "#f59e0b" }}>
            Click "Compute priors" first to enable this view.
          </span>
        )}
      </div>

      {/* Errors */}
      {error && view === "observed" && (
        <div style={{ ...PANEL, marginBottom: "16px", borderColor: "rgba(239,68,68,0.4)", color: "#fca5a5", fontSize: "13px" }}>
          {error.includes("404") || error.includes("No topic tree")
            ? "No topic tree built yet. Click \"Rebuild topic tree\" first."
            : error}
        </div>
      )}

      {/* Matrix */}
      {matrix && displayedTopics.length > 0 && (
        <div style={{ ...PANEL, overflowX: "auto" }}>
          <table style={{ borderCollapse: "collapse", width: "100%", fontSize: "12px" }}>
            <thead>
              <tr>
                <th style={{
                  textAlign: "left", padding: "10px 12px",
                  position: "sticky", left: 0, background: "#0a1424",
                  borderBottom: "1px solid rgba(255,255,255,0.08)",
                  color: "#64748b", fontWeight: 500, fontSize: "10px",
                  textTransform: "uppercase", letterSpacing: "0.07em",
                }}>
                  Agent
                </th>
                {displayedTopics.map((t) => (
                  <th
                    key={t.id}
                    style={{
                      padding: "10px 8px",
                      borderBottom: "1px solid rgba(255,255,255,0.08)",
                      borderLeft: "1px solid rgba(255,255,255,0.04)",
                      color: "#cbd5e1", fontWeight: 600, textAlign: "center",
                      minWidth: "90px", maxWidth: "150px",
                      verticalAlign: "bottom",
                    }}
                    title={t.label}
                  >
                    <div style={{ fontSize: "11px", lineHeight: 1.3 }}>{t.label}</div>
                    <div style={{ fontSize: "9px", color: "#475569", marginTop: "3px" }}>
                      {`${t.member_post_ids.length} posts`}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.agents.map((a) => (
                <tr key={a.id}>
                  <td style={{
                    padding: "8px 12px",
                    position: "sticky", left: 0, background: "#0a1424",
                    borderBottom: "1px solid rgba(255,255,255,0.04)",
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: a.avatar_color }} />
                      <div>
                        <div style={{ color: "#e2e8f0", fontWeight: 500 }}>{a.name}</div>
                        <div style={{ fontSize: "10px", color: "#475569" }}>{a.domain || a.role}</div>
                      </div>
                    </div>
                  </td>
                  {displayedTopics.map((t) => {
                    const cv = cellValue(a.id, t.id);
                    if (!cv) {
                      return (
                        <td key={t.id} style={{ textAlign: "center", padding: "10px 6px", color: "#334155", fontFamily: "monospace", borderLeft: "1px solid rgba(255,255,255,0.04)" }}>
                          —
                        </td>
                      );
                    }
                    const lightText = view === "drift" ? Math.abs(cv.display) > 0.25 : cv.display > 0.55;
                    return (
                      <td key={t.id} style={{
                        textAlign: "center", padding: "10px 6px",
                        background: cv.bg,
                        color: lightText ? "#060d1a" : "#cbd5e1",
                        fontFamily: "monospace", fontWeight: 600,
                        borderLeft: "1px solid rgba(255,255,255,0.04)",
                        borderBottom: "1px solid rgba(255,255,255,0.04)",
                      }} title={cv.tooltip}>
                        {view === "drift"
                          ? `${cv.display >= 0 ? "+" : ""}${cv.display.toFixed(2)}`
                          : cv.display.toFixed(2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>

          {/* Legend */}
          <div style={{ display: "flex", alignItems: "center", gap: "12px", marginTop: "14px", fontSize: "11px", color: "#475569", flexWrap: "wrap" }}>
            <span>
              {view === "observed" && "Interest (activity-based):"}
              {view === "prior" && "Interest (persona-derived prior):"}
              {view === "drift" && "Drift (observed − prior):"}
            </span>
            <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
              {(view === "drift" ? [-1, -0.5, 0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1]).map((v) => (
                <div key={v} style={{
                  width: "34px", height: "12px",
                  background: view === "drift" ? driftColor(v) : heatColor(v),
                  fontSize: "9px",
                  color: (view === "drift" ? Math.abs(v) > 0.25 : v > 0.55) ? "#060d1a" : "#cbd5e1",
                  display: "flex", alignItems: "center", justifyContent: "center",
                }}>
                  {v >= 0 && view === "drift" ? "+" : ""}{v.toFixed(2)}
                </div>
              ))}
            </div>
            <span style={{ marginLeft: "auto" }}>
              {view === "observed" && "sigmoid(1·engaged + 0.5·authored − 0.3·skipped); prelude folded in at 0.25×"}
              {view === "prior" && "LLM-rated 0..1 given persona alone"}
              {view === "drift" && "+ = engaged more than persona predicted, − = less"}
            </span>
          </div>
        </div>
      )}

      {/* Research metrics panel */}
      {metrics && (
        <div style={{ ...PANEL, marginTop: "16px" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: "16px", marginBottom: "14px", flexWrap: "wrap" }}>
            <h2 style={{ margin: 0, fontSize: "17px", color: "#00d4ff", fontWeight: 600 }}>Research metrics</h2>
            <span style={{ fontSize: "11px", color: "#64748b" }}>
              Avg fidelity {metrics.avg_persona_fidelity.toFixed(3)} ·
              Avg skip {(metrics.avg_skip_rate * 100).toFixed(1)}% ·
              Avg quality {metrics.avg_post_quality !== null ? metrics.avg_post_quality.toFixed(3) : "—"} ·
              {metrics.quality_scored_posts}/{metrics.total_posts} posts judged
            </span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 2fr) minmax(0, 1fr)", gap: "16px" }}>
            {/* Per-agent table */}
            <div>
              <div style={{ fontSize: "10px", color: "#475569", textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: "8px" }}>Per agent</div>
              <table style={{ borderCollapse: "collapse", width: "100%", fontSize: "12px" }}>
                <thead>
                  <tr style={{ color: "#64748b", fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                    <th style={{ textAlign: "left", padding: "6px 8px" }}>Agent</th>
                    <th style={{ textAlign: "right", padding: "6px 8px" }} title="cosine(prior_row, observed_row) — 1.0 means the recommender's model matches the persona">Fidelity</th>
                    <th style={{ textAlign: "right", padding: "6px 8px" }} title="skipped / (engaged + skipped) across all recommendations">Skip %</th>
                    <th style={{ textAlign: "right", padding: "6px 8px" }} title="mean LLM-judge quality of posts authored by this agent">Quality</th>
                    <th style={{ textAlign: "right", padding: "6px 8px", color: "#334155" }}>Posts</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.per_agent.map((a) => (
                    <tr key={a.agent_id} style={{ borderTop: "1px solid rgba(255,255,255,0.04)" }}>
                      <td style={{ padding: "6px 8px" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                          <div style={{ width: "6px", height: "6px", borderRadius: "50%", background: a.avatar_color }} />
                          <span>{a.agent_name}</span>
                        </div>
                      </td>
                      <td style={{ textAlign: "right", padding: "6px 8px", fontFamily: "monospace", color: a.persona_fidelity > 0.7 ? "#10b981" : a.persona_fidelity > 0.4 ? "#f59e0b" : "#ef4444" }}>
                        {a.persona_fidelity.toFixed(3)}
                      </td>
                      <td style={{ textAlign: "right", padding: "6px 8px", fontFamily: "monospace", color: a.skip_rate > 0.6 ? "#ef4444" : a.skip_rate > 0.3 ? "#f59e0b" : "#10b981" }}>
                        {(a.skip_rate * 100).toFixed(1)}%
                      </td>
                      <td style={{ textAlign: "right", padding: "6px 8px", fontFamily: "monospace", color: a.avg_post_quality !== null ? (a.avg_post_quality > 0.7 ? "#10b981" : a.avg_post_quality > 0.4 ? "#f59e0b" : "#ef4444") : "#334155" }}>
                        {a.avg_post_quality !== null ? a.avg_post_quality.toFixed(3) : "—"}
                      </td>
                      <td style={{ textAlign: "right", padding: "6px 8px", fontFamily: "monospace", color: "#475569" }}>
                        {a.posts_authored}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Per-topic surprise */}
            <div>
              <div style={{ fontSize: "10px", color: "#475569", textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: "8px" }}>Most surprising topics</div>
              <table style={{ borderCollapse: "collapse", width: "100%", fontSize: "12px" }}>
                <thead>
                  <tr style={{ color: "#64748b", fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                    <th style={{ textAlign: "left", padding: "6px 8px" }}>Topic</th>
                    <th style={{ textAlign: "right", padding: "6px 8px" }} title="mean |observed − prior| across all agents">Δ</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.per_topic_surprise.map((t) => (
                    <tr key={t.topic_id} style={{ borderTop: "1px solid rgba(255,255,255,0.04)" }}>
                      <td style={{ padding: "6px 8px", color: "#cbd5e1" }}>{t.label}</td>
                      <td style={{ textAlign: "right", padding: "6px 8px", fontFamily: "monospace", color: "#00d4ff" }}>
                        {t.mean_abs_drift.toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div style={{ marginTop: "12px", fontSize: "10px", color: "#475569", lineHeight: 1.6 }}>
            <b style={{ color: "#64748b" }}>Fidelity</b> = cosine similarity between the persona-derived prior and the observed activity model.
            High fidelity means the recommender modelled this agent's persona accurately.{" "}
            <b style={{ color: "#64748b" }}>Skip rate</b> measures how often the agent rejected the recommender's suggestion.{" "}
            <b style={{ color: "#64748b" }}>Quality</b> is an LLM judge's 0–1 rating on specificity, relevance, and substance per post.
          </div>
        </div>
      )}

      {/* (per-agent topic tree now lives at the top of the page) */}

      {/* Feature 2: per-agent action documents */}
      {reports.length > 0 && (
        <div style={{ ...PANEL, marginTop: "16px" }}>
          <h2 style={{ margin: "0 0 12px", fontSize: "17px", color: "#00d4ff", fontWeight: 600 }}>Agent action reports</h2>
          <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", marginBottom: "12px" }}>
            {reports.map((r) => (
              <button key={r.agent_id} onClick={() => setOpenReportId(r.agent_id)}
                style={{ ...HEADER_BTN, background: openReportId === r.agent_id ? "#00d4ff" : "transparent", color: openReportId === r.agent_id ? "#060d1a" : "#94a3b8", border: openReportId === r.agent_id ? "none" : "1px solid rgba(255,255,255,0.12)" }}>
                {r.name}
              </button>
            ))}
          </div>
          {reports.filter((r) => r.agent_id === openReportId).map((r) => (
            <pre key={r.agent_id} style={{
              background: "#0a1424", border: "1px solid rgba(255,255,255,0.08)", borderRadius: "8px",
              padding: "14px 16px", color: "#cbd5e1", fontSize: "11.5px", lineHeight: 1.55,
              whiteSpace: "pre-wrap", wordBreak: "break-word", maxHeight: "440px", overflowY: "auto", margin: 0,
            }}>
              {r.text}
            </pre>
          ))}
          <div style={{ marginTop: "10px", fontSize: "10px", color: "#475569" }}>
            Reports are also written to <code>backend/agent_reports/&lt;session&gt;/</code> as text files.
          </div>
        </div>
      )}

      {priorError && view !== "observed" && (
        <div style={{ ...PANEL, marginTop: "12px", borderColor: "rgba(245,158,11,0.4)", color: "#fcd34d", fontSize: "12px" }}>
          Persona priors not yet computed for this session. Click <b>Compute priors</b> above.
        </div>
      )}

      {matrix && displayedTopics.length === 0 && !error && view === "observed" && (
        <div style={{ ...PANEL, color: "#64748b", fontSize: "13px" }}>
          No clustered topics yet. Click <b>Rebuild topic tree</b> above.
        </div>
      )}
    </div>
  );
}
