"use client";
import { useState } from "react";
import { PTYPE_COLORS } from "./PostCard";

interface TreeNode {
  id: string;
  ptype: string;
  name: string;
  description: string;
  achievers?: TreeNode[];
  sim_tick?: number;
}

interface TreeNodeProps {
  node: TreeNode;
  depth?: number;
  selectedId?: string;
  onSelect: (node: TreeNode) => void;
}

function TreeNodeItem({ node, depth = 0, selectedId, onSelect }: TreeNodeProps) {
  const [expanded, setExpanded] = useState(depth < 2);
  const color = PTYPE_COLORS[node.ptype] ?? "#64748b";
  const hasChildren = (node.achievers?.length ?? 0) > 0;
  const isSelected = node.id === selectedId;

  return (
    <div>
      <div
        className="flex items-center gap-1 py-1 px-2 rounded cursor-pointer hover:bg-white/5 transition-colors"
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={() => onSelect(node)}
      >
        {hasChildren ? (
          <button
            className="w-4 h-4 flex items-center justify-center text-gray-400 hover:text-white shrink-0"
            onClick={(e) => { e.stopPropagation(); setExpanded(!expanded); }}
          >
            {expanded ? "▾" : "▸"}
          </button>
        ) : (
          <span className="w-4 shrink-0" />
        )}
        <span
          className="text-xs px-1.5 py-0.5 rounded font-medium shrink-0"
          style={{ background: color, color: "#fff", fontSize: "10px" }}
        >
          {node.ptype}
        </span>
        <span
          className="text-sm truncate"
          style={{ color: isSelected ? color : "#e2e8f0" }}
        >
          {node.name}
        </span>
      </div>
      {expanded && hasChildren && (
        <div>
          {node.achievers!.map((child) => (
            <TreeNodeItem
              key={child.id}
              node={child}
              depth={depth + 1}
              selectedId={selectedId}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  );
}

interface TreeViewProps {
  roots: TreeNode[];
  selectedId?: string;
  onSelect: (node: TreeNode) => void;
}

export default function TreeView({ roots, selectedId, onSelect }: TreeViewProps) {
  if (!roots || roots.length === 0) {
    return <div style={{ color: "#475569", fontSize: "13px", padding: "12px" }}>No topics yet. Start a session.</div>;
  }
  return (
    <div className="overflow-auto">
      {roots.map((root, i) => (
        <div key={root.id} style={{ marginBottom: i < roots.length - 1 ? "8px" : 0 }}>
          {roots.length > 1 && (
            <div style={{ fontSize: "10px", color: "#475569", padding: "4px 8px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Topic {i + 1}
            </div>
          )}
          <TreeNodeItem node={root} selectedId={selectedId} onSelect={onSelect} />
          {roots.length > 1 && i < roots.length - 1 && (
            <div style={{ height: "1px", background: "rgba(255,255,255,0.05)", margin: "6px 0" }} />
          )}
        </div>
      ))}
    </div>
  );
}
