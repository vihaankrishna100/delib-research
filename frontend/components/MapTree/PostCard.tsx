"use client";

const PTYPE_COLORS: Record<string, string> = {
  mission: "#1e3a5f",
  stakeholder: "#64748b",
  goal: "#10b981",
  barrier: "#ef4444",
  cause: "#f97316",
  solution: "#00d4ff",
  abstraction: "#8b5cf6",
  analogy: "#6366f1",
  inspiration: "#f59e0b",
  question: "#ec4899",
  answer: "#38bdf8",
  pro: "#22c55e",
  con: "#dc2626",
  part: "#14b8a6",
  failure_mode: "#f43f5e",
};

interface Post {
  id: string;
  ptype: string;
  name: string;
  description: string;
  purpose_id?: string;
  author_agent_id?: string;
  content_mode?: string;
  sim_tick?: number;
}

interface PostCardProps {
  post: Post;
  selected?: boolean;
  onClick?: () => void;
}

export default function PostCard({ post, selected, onClick }: PostCardProps) {
  const color = PTYPE_COLORS[post.ptype] ?? "#64748b";
  return (
    <div
      onClick={onClick}
      className="cursor-pointer rounded-lg p-3 border transition-all"
      style={{
        borderColor: selected ? color : "rgba(255,255,255,0.08)",
        background: selected ? `${color}22` : "rgba(255,255,255,0.03)",
        boxShadow: selected ? `0 0 0 1px ${color}` : "none",
      }}
    >
      <div className="flex items-center gap-2 mb-1">
        <span
          className="text-xs font-semibold px-2 py-0.5 rounded uppercase tracking-wide"
          style={{ background: color, color: "#fff" }}
        >
          {post.ptype}
        </span>
        {post.content_mode === "self_written" && (
          <span className="text-xs text-gray-500">self-written</span>
        )}
        {post.sim_tick !== undefined && (
          <span className="text-xs text-gray-600 ml-auto">t{post.sim_tick}</span>
        )}
      </div>
      <div className="text-sm font-medium text-white">{post.name}</div>
      <div className="text-xs text-gray-400 mt-1 line-clamp-2">{post.description}</div>
    </div>
  );
}

export { PTYPE_COLORS };
