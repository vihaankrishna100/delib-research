"use client";
import { useState, useEffect, useCallback } from "react";
import TreeView from "@/components/MapTree/TreeView";
import PostCard, { PTYPE_COLORS } from "@/components/MapTree/PostCard";
import { getSessions, getTree, getPostContext, getAgents, getEvents } from "@/lib/api";

interface Post {
  id: string;
  ptype: string;
  name: string;
  description: string;
  purpose_id?: string;
  author_agent_id?: string;
  content_mode?: string;
  sim_tick?: number;
  achievers?: Post[];
}

interface Session { id: string; status: string; tick_count: number; }
interface Agent { id: string; name: string; role: string; domain: string; avatar_color: string; }
interface LiveEvent {
  id: string;
  agent_id?: string;
  event_type: string;
  metadata?: Record<string, unknown>;
  sim_tick: number;
}

const STATUS_COLOR: Record<string, string> = {
  running: "#10b981", idle: "#64748b", paused: "#f59e0b", stopped: "#475569",
};

export default function MapPage() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [roots, setRoots] = useState<Post[]>([]);
  const [allPosts, setAllPosts] = useState<Record<string, Post>>({});
  const [selectedId, setSelectedId] = useState<string>("");
  const [contextChain, setContextChain] = useState<Post[]>([]);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [events, setEvents] = useState<LiveEvent[]>([]);

  // Load sessions + agents once
  useEffect(() => {
    getSessions()
      .then((s: Session[]) => {
        setSessions(s);
        if (s.length > 0) setActiveSessionId(s[0].id);
      })
      .catch(() => {});
    getAgents().then(setAgents).catch(() => {});
  }, []);

  // Poll tree + events every 3s when session is active
  const refresh = useCallback(() => {
    if (!activeSessionId) return;
    getTree(activeSessionId)
      .then((data: { roots: Post[]; all_posts: Record<string, Post> }) => {
        setRoots(data.roots ?? []);
        setAllPosts(data.all_posts ?? {});
      })
      .catch(() => {});
    getEvents(activeSessionId, 40)
      .then((evts: LiveEvent[]) => setEvents(evts))
      .catch(() => {});
  }, [activeSessionId]);

  useEffect(() => {
    refresh();
    const iv = setInterval(refresh, 3000);
    return () => clearInterval(iv);
  }, [refresh]);

  const handleSelectNode = (node: Post) => {
    setSelectedId(node.id);
    getPostContext(node.id).then(setContextChain).catch(() => {});
  };

  const agentMap = Object.fromEntries(agents.map((a) => [a.id, a]));

  // Always derive selected post + live children from allPosts (never stale)
  const selectedPost = selectedId ? allPosts[selectedId] : null;
  const liveChildren = selectedId
    ? Object.values(allPosts).filter((p) => p.purpose_id === selectedId)
    : [];

  const activeSession = sessions.find((s) => s.id === activeSessionId);

  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "280px 1fr 280px",
      height: "calc(100vh - 49px)",
      overflow: "hidden",
      background: "#060d1a",
    }}>

      {/* ── Left: topic tree ── */}
      <div style={{
        borderRight: "1px solid rgba(255,255,255,0.07)",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}>
        {/* Session selector */}
        <div style={{ padding: "10px 12px", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
          <select
            value={activeSessionId}
            onChange={(e) => { setActiveSessionId(e.target.value); setSelectedId(""); setContextChain([]); }}
            style={{
              width: "100%",
              background: "#0f172a",
              color: "#e2e8f0",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: "6px",
              padding: "6px 8px",
              fontSize: "12px",
            }}
          >
            <option value="">— select session —</option>
            {sessions.map((s) => (
              <option key={s.id} value={s.id}>
                {s.id.slice(0, 8)} · {s.status} · t{s.tick_count}
              </option>
            ))}
          </select>
          {activeSession && (
            <div style={{ marginTop: "6px", display: "flex", alignItems: "center", gap: "6px", fontSize: "11px" }}>
              <div style={{ width: "6px", height: "6px", borderRadius: "50%", background: STATUS_COLOR[activeSession.status] ?? "#64748b" }} />
              <span style={{ color: "#64748b" }}>{activeSession.status}</span>
              <span style={{ color: "#334155", marginLeft: "auto" }}>tick {activeSession.tick_count}</span>
            </div>
          )}
        </div>

        {/* Tree */}
        <div style={{ flex: 1, overflowY: "auto", padding: "8px 4px" }}>
          {!activeSessionId ? (
            <div style={{ padding: "16px 12px", color: "#334155", fontSize: "13px" }}>
              Select a session to view the map.
            </div>
          ) : roots.length === 0 ? (
            <div style={{ padding: "16px 12px", color: "#334155", fontSize: "13px" }}>
              No topics yet — start the simulation.
            </div>
          ) : (
            <TreeView roots={roots} selectedId={selectedId} onSelect={handleSelectNode} />
          )}
        </div>
      </div>

      {/* ── Center: post detail ── */}
      <div style={{ overflowY: "auto", padding: "28px 32px" }}>
        {!selectedPost ? (
          <div style={{
            height: "100%",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            color: "#1e293b",
            gap: "10px",
          }}>
            <div style={{ fontSize: "52px" }}>🗺️</div>
            <div style={{ fontSize: "14px" }}>Click any node in the tree to explore</div>
          </div>
        ) : (
          <>
            {/* Breadcrumb */}
            {contextChain.length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "4px", marginBottom: "20px", fontSize: "12px" }}>
                {contextChain.map((c, i) => (
                  <span key={c.id} style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                    {i > 0 && <span style={{ color: "#334155" }}>›</span>}
                    <span
                      onClick={() => handleSelectNode(c)}
                      style={{ color: PTYPE_COLORS[c.ptype] ?? "#64748b", cursor: "pointer", opacity: c.id === selectedId ? 1 : 0.6 }}
                    >
                      {c.name}
                    </span>
                  </span>
                ))}
              </div>
            )}

            {/* Post header */}
            <div style={{ marginBottom: "20px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px" }}>
                <span style={{
                  background: PTYPE_COLORS[selectedPost.ptype] ?? "#64748b",
                  color: "#fff",
                  padding: "3px 10px",
                  borderRadius: "4px",
                  fontSize: "11px",
                  fontWeight: 700,
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                }}>
                  {selectedPost.ptype}
                </span>
                <span style={{ fontSize: "11px", color: "#475569" }}>
                  tick {selectedPost.sim_tick ?? 0} · {selectedPost.content_mode ?? ""}
                </span>
              </div>
              <h2 style={{ fontSize: "20px", fontWeight: 600, color: "#f1f5f9", marginBottom: "10px", lineHeight: 1.3 }}>
                {selectedPost.name}
              </h2>
              <p style={{ color: "#94a3b8", lineHeight: 1.65, fontSize: "14px" }}>
                {selectedPost.description}
              </p>
              {selectedPost.author_agent_id && agentMap[selectedPost.author_agent_id] && (
                <div style={{ marginTop: "10px", fontSize: "12px", color: "#475569" }}>
                  By{" "}
                  <span style={{ color: agentMap[selectedPost.author_agent_id].avatar_color, fontWeight: 600 }}>
                    {agentMap[selectedPost.author_agent_id].name}
                  </span>
                  {" "}· {agentMap[selectedPost.author_agent_id].domain}
                </div>
              )}
            </div>

            {/* Live children */}
            {liveChildren.length > 0 && (
              <div>
                <div style={{ fontSize: "11px", color: "#475569", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "10px" }}>
                  {liveChildren.length} {liveChildren.length === 1 ? "child" : "children"}
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                  {liveChildren
                    .sort((a, b) => (a.sim_tick ?? 0) - (b.sim_tick ?? 0))
                    .map((child) => (
                      <PostCard
                        key={child.id}
                        post={child}
                        selected={child.id === selectedId}
                        onClick={() => handleSelectNode(child)}
                      />
                    ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* ── Right: agents + events ── */}
      <div style={{
        borderLeft: "1px solid rgba(255,255,255,0.07)",
        overflowY: "auto",
        padding: "12px",
      }}>
        {/* Agents */}
        <div style={{ fontSize: "10px", color: "#475569", textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: "8px" }}>
          Agents
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "3px", marginBottom: "20px" }}>
          {agents.map((a) => (
            <div key={a.id} style={{
              display: "flex", alignItems: "center", gap: "7px",
              padding: "5px 7px", borderRadius: "5px",
              background: "rgba(255,255,255,0.025)",
            }}>
              <div style={{ width: "7px", height: "7px", borderRadius: "50%", background: a.avatar_color, flexShrink: 0 }} />
              <span style={{ fontSize: "12px", color: "#cbd5e1", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {a.name}
              </span>
              <span style={{ fontSize: "9px", color: "#475569", textTransform: "uppercase" }}>{a.role.slice(0, 4)}</span>
            </div>
          ))}
        </div>

        {/* Live events */}
        <div style={{ fontSize: "10px", color: "#475569", textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: "8px" }}>
          Live Events
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "3px" }}>
          {events.length === 0 ? (
            <div style={{ color: "#334155", fontSize: "12px" }}>No events yet.</div>
          ) : events.map((e) => {
            const agent = e.agent_id ? agentMap[e.agent_id] : null;
            const isCreation = Boolean(e.metadata?.new_post_name);
            return (
              <div key={e.id} style={{
                padding: "5px 7px",
                borderRadius: "4px",
                background: isCreation ? "rgba(0,212,255,0.04)" : "rgba(255,255,255,0.02)",
                borderLeft: isCreation ? "2px solid rgba(0,212,255,0.3)" : "2px solid transparent",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "11px" }}>
                  {agent && <span style={{ color: agent.avatar_color, fontWeight: 600, flexShrink: 0 }}>{agent.name}</span>}
                  <span style={{ color: "#64748b", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {e.event_type}
                  </span>
                  <span style={{ color: "#334155", fontSize: "10px", flexShrink: 0 }}>t{e.sim_tick}</span>
                </div>
                {e.metadata?.new_post_name ? (
                  <div style={{ color: "#64748b", fontSize: "10px", marginTop: "2px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {String(e.metadata.new_post_name).slice(0, 55)}
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
