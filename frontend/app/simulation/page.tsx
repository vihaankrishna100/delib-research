"use client";
import { useState, useEffect } from "react";
import { getSessions, createSession, startSession, pauseSession, resumeSession, stopSession, getSession } from "@/lib/api";

interface Session {
  id: string;
  status: string;
  tick_count: number;
  config?: Record<string, unknown>;
  mission_id?: string;
}

export default function SimulationPage() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const [activeSession, setActiveSession] = useState<Session | null>(null);
  const [missionName, setMissionName] = useState("Improve urban public transit accessibility");
  const [missionDesc, setMissionDesc] = useState("Design a deliberation process to identify barriers and solutions for making urban public transit more accessible, affordable, and efficient for all residents.");
  const [strategy, setStrategy] = useState("ab_rotate");
  const [agentMode, setAgentMode] = useState("llm");
  const [ticks, setTicks] = useState(20);
  const [preludeTicks, setPreludeTicks] = useState(5);
  const [agentSampleSize, setAgentSampleSize] = useState(0);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState("");

  const refresh = async () => {
    const s: Session[] = await getSessions().catch(() => []);
    setSessions(s);
    if (activeId) {
      const current = await getSession(activeId).catch(() => null);
      if (current) setActiveSession(current);
    }
  };

  useEffect(() => {
    refresh();
    const iv = setInterval(refresh, 2000);
    return () => clearInterval(iv);
  }, [activeId]);

  const handleCreate = async () => {
    setLoading(true);
    try {
      const s = await createSession({
        mission_name: missionName,
        mission_description: missionDesc,
        config: { recommendation_strategy: strategy, num_ticks: ticks, prelude_ticks: preludeTicks, agent_sample_size: agentSampleSize, agent_mode: agentMode },
      });
      setSessions((prev) => [s, ...prev]);
      setActiveId(s.id);
      setActiveSession(s);
      setMsg("Session created.");
    } catch (e) {
      setMsg("Error creating session.");
    }
    setLoading(false);
  };

  const handleStart = async () => {
    if (!activeId) return;
    setLoading(true);
    await startSession(activeId, ticks).catch(() => setMsg("Error starting."));
    setMsg("Started.");
    setLoading(false);
  };

  const handlePause = async () => {
    if (!activeId) return;
    await pauseSession(activeId).catch(() => {});
    setMsg("Paused.");
    refresh();
  };

  const handleResume = async () => {
    if (!activeId) return;
    await resumeSession(activeId, ticks).catch(() => {});
    setMsg("Resumed.");
  };

  const handleStop = async () => {
    if (!activeId) return;
    await stopSession(activeId).catch(() => {});
    setMsg("Stopped.");
    refresh();
  };

  const statusColor: Record<string, string> = {
    idle: "#64748b", running: "#10b981", paused: "#f59e0b", stopped: "#ef4444"
  };

  return (
    <div style={{ maxWidth: "800px", margin: "0 auto", padding: "32px 24px" }}>
      <h1 style={{ fontSize: "24px", fontWeight: 700, color: "#00d4ff", marginBottom: "24px" }}>
        Simulation Control
      </h1>

      {/* Create new session */}
      <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: "10px", padding: "20px", marginBottom: "24px" }}>
        <h2 style={{ fontSize: "15px", fontWeight: 600, marginBottom: "16px", color: "#e2e8f0" }}>New Session</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          <div>
            <label style={{ fontSize: "12px", color: "#64748b", display: "block", marginBottom: "4px" }}>Mission</label>
            <input
              value={missionName}
              onChange={(e) => setMissionName(e.target.value)}
              style={{ width: "100%", background: "#0f172a", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "6px", padding: "8px 12px", color: "#e2e8f0", fontSize: "14px", boxSizing: "border-box" }}
            />
          </div>
          <div>
            <label style={{ fontSize: "12px", color: "#64748b", display: "block", marginBottom: "4px" }}>Mission Description</label>
            <textarea
              value={missionDesc}
              onChange={(e) => setMissionDesc(e.target.value)}
              rows={3}
              style={{ width: "100%", background: "#0f172a", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "6px", padding: "8px 12px", color: "#e2e8f0", fontSize: "13px", resize: "vertical", boxSizing: "border-box" }}
            />
          </div>
          <div style={{ display: "flex", gap: "16px" }}>
            <div style={{ flex: 1 }}>
              <label style={{ fontSize: "12px", color: "#64748b", display: "block", marginBottom: "4px" }}>Recommendation Strategy</label>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                style={{ width: "100%", background: "#0f172a", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "6px", padding: "8px 12px", color: "#e2e8f0", fontSize: "14px" }}
              >
                <option value="ab_rotate">A/B Rotate (all 3)</option>
                <option value="random">Random</option>
                <option value="priority">Priority (gap-filling)</option>
                <option value="multi_agent">Multi-Agent LLM</option>
              </select>
            </div>
            <div style={{ width: "120px" }}>
              <label style={{ fontSize: "12px", color: "#64748b", display: "block", marginBottom: "4px" }}>Ticks</label>
              <input
                type="number"
                value={ticks}
                onChange={(e) => setTicks(Number(e.target.value))}
                min={1} max={100}
                style={{ width: "100%", background: "#0f172a", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "6px", padding: "8px 12px", color: "#e2e8f0", fontSize: "14px", boxSizing: "border-box" }}
              />
            </div>
            <div style={{ width: "120px" }}>
              <label
                title="A short warm-start deliberation that runs before the main simulation. Its activity seeds the persona table at low weight, avoiding a cold start."
                style={{ fontSize: "12px", color: "#64748b", display: "block", marginBottom: "4px" }}
              >
                Prelude ticks
              </label>
              <input
                type="number"
                value={preludeTicks}
                onChange={(e) => setPreludeTicks(Number(e.target.value))}
                min={0} max={20}
                style={{ width: "100%", background: "#0f172a", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "6px", padding: "8px 12px", color: "#e2e8f0", fontSize: "14px", boxSizing: "border-box" }}
              />
            </div>
          </div>
          <div style={{ fontSize: "11px", color: "#475569", marginTop: "-4px" }}>
            Prelude runs {preludeTicks} low-weight warm-start tick{preludeTicks === 1 ? "" : "s"} before the {ticks} main ticks — it&apos;s visualized and seeds the persona table to avoid a cold start.
          </div>
          <div style={{ display: "flex", gap: "16px", alignItems: "flex-end" }}>
            <div style={{ width: "260px" }}>
              <label
                title="LLM: agents decide skip with free judgment (naturalistic, less consistent). Synthetic: each agent has a hidden ground-truth persona model that decides skip deterministically — consistent, and lets us train + score the decision-tree user model."
                style={{ fontSize: "12px", color: "#64748b", display: "block", marginBottom: "4px" }}
              >
                Agent decision mode
              </label>
              <select
                value={agentMode}
                onChange={(e) => setAgentMode(e.target.value)}
                style={{ width: "100%", background: "#0f172a", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "6px", padding: "8px 12px", color: "#e2e8f0", fontSize: "14px" }}
              >
                <option value="llm">LLM judgment (naturalistic)</option>
                <option value="synthetic">Synthetic ground-truth (consistent)</option>
              </select>
            </div>
            <div style={{ fontSize: "11px", color: "#475569", paddingBottom: "9px" }}>
              {agentMode === "synthetic"
                ? "Deterministic skip from a hidden persona model → trains & scores the decision-tree user model."
                : "Free LLM skip decisions — the original behavior."}
            </div>
          </div>
          <div style={{ display: "flex", gap: "16px", alignItems: "flex-end" }}>
            <div style={{ width: "180px" }}>
              <label
                title="Run with only the first N specialists (by name) to save time and API calls in experiments. 0 = all 8 specialists."
                style={{ fontSize: "12px", color: "#64748b", display: "block", marginBottom: "4px" }}
              >
                Agents in run (0 = all)
              </label>
              <input
                type="number"
                value={agentSampleSize}
                onChange={(e) => setAgentSampleSize(Number(e.target.value))}
                min={0} max={8}
                style={{ width: "100%", background: "#0f172a", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "6px", padding: "8px 12px", color: "#e2e8f0", fontSize: "14px", boxSizing: "border-box" }}
              />
            </div>
            <div style={{ fontSize: "11px", color: "#475569", paddingBottom: "9px" }}>
              {agentSampleSize > 0
                ? `Experiment mode: only ${agentSampleSize} specialist${agentSampleSize === 1 ? "" : "s"} will act — faster, fewer API calls.`
                : "All 8 specialists will participate."}
            </div>
          </div>
          <button
            onClick={handleCreate}
            disabled={loading}
            style={{ background: "#6366f1", color: "#fff", border: "none", borderRadius: "6px", padding: "10px 20px", cursor: "pointer", fontSize: "14px", fontWeight: 600, alignSelf: "flex-start" }}
          >
            Create Session
          </button>
        </div>
      </div>

      {/* Session controls */}
      <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: "10px", padding: "20px", marginBottom: "24px" }}>
        <h2 style={{ fontSize: "15px", fontWeight: 600, marginBottom: "16px", color: "#e2e8f0" }}>Session Control</h2>
        <div style={{ marginBottom: "12px" }}>
          <select
            value={activeId}
            onChange={(e) => setActiveId(e.target.value)}
            style={{ width: "100%", background: "#0f172a", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "6px", padding: "8px 12px", color: "#e2e8f0", fontSize: "14px" }}
          >
            <option value="">Select session</option>
            {sessions.map((s) => (
              <option key={s.id} value={s.id}>
                {s.id.slice(0, 8)} — {s.status} — tick {s.tick_count}
              </option>
            ))}
          </select>
        </div>
        {activeSession && (
          <div style={{ marginBottom: "16px", padding: "12px", background: "#0f172a", borderRadius: "6px", fontSize: "13px" }}>
            <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
              <span>Status: <span style={{ color: statusColor[activeSession.status] ?? "#fff", fontWeight: 600 }}>{activeSession.status}</span></span>
              <span>Ticks completed: <span style={{ color: "#00d4ff" }}>{activeSession.tick_count}</span></span>
              {activeSession.config && (
                <span>Strategy: <span style={{ color: "#a78bfa" }}>{String((activeSession.config as Record<string, unknown>).recommendation_strategy)}</span></span>
              )}
            </div>
          </div>
        )}
        <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
          <button onClick={handleStart} disabled={loading || !activeId || activeSession?.status === "running"}
            style={{ background: "#10b981", color: "#fff", border: "none", borderRadius: "6px", padding: "8px 16px", cursor: "pointer", fontSize: "13px", fontWeight: 600 }}>
            ▶ Start
          </button>
          <button onClick={handlePause} disabled={!activeId || activeSession?.status !== "running"}
            style={{ background: "#f59e0b", color: "#fff", border: "none", borderRadius: "6px", padding: "8px 16px", cursor: "pointer", fontSize: "13px", fontWeight: 600 }}>
            ⏸ Pause
          </button>
          <button onClick={handleResume} disabled={!activeId || activeSession?.status === "running"}
            style={{ background: "#6366f1", color: "#fff", border: "none", borderRadius: "6px", padding: "8px 16px", cursor: "pointer", fontSize: "13px", fontWeight: 600 }}>
            ▶▶ Resume
          </button>
          <button onClick={handleStop} disabled={!activeId}
            style={{ background: "#ef4444", color: "#fff", border: "none", borderRadius: "6px", padding: "8px 16px", cursor: "pointer", fontSize: "13px", fontWeight: 600 }}>
            ■ Stop
          </button>
        </div>
        {msg && <div style={{ marginTop: "12px", fontSize: "13px", color: "#64748b" }}>{msg}</div>}
      </div>

      {/* Session list */}
      <div>
        <h2 style={{ fontSize: "15px", fontWeight: 600, marginBottom: "12px", color: "#e2e8f0" }}>All Sessions</h2>
        {sessions.length === 0 ? (
          <div style={{ color: "#475569", fontSize: "13px" }}>No sessions yet.</div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            {sessions.map((s) => (
              <div
                key={s.id}
                onClick={() => setActiveId(s.id)}
                style={{ padding: "10px 14px", background: activeId === s.id ? "rgba(99,102,241,0.12)" : "rgba(255,255,255,0.03)", border: `1px solid ${activeId === s.id ? "#6366f1" : "rgba(255,255,255,0.08)"}`, borderRadius: "6px", cursor: "pointer", display: "flex", gap: "16px", fontSize: "13px" }}
              >
                <span style={{ color: "#94a3b8", fontFamily: "monospace" }}>{s.id.slice(0, 8)}</span>
                <span style={{ color: statusColor[s.status] ?? "#fff" }}>{s.status}</span>
                <span style={{ color: "#64748b" }}>t{s.tick_count}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
