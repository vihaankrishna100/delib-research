const BASE = "http://localhost:8001";

export async function apiFetch(path: string, options?: RequestInit) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json();
}

// Sessions
export const getSessions = () => apiFetch("/simulation/sessions");
export const getSession = (id: string) => apiFetch(`/simulation/sessions/${id}`);
export const createSession = (body: {
  mission_name: string;
  mission_description: string;
  config?: { recommendation_strategy?: string; num_ticks?: number; prelude_ticks?: number; agent_sample_size?: number; agent_mode?: string };
}) => apiFetch("/simulation/sessions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const startSession = (id: string, ticks = 20) =>
  apiFetch(`/simulation/sessions/${id}/start?ticks=${ticks}`, { method: "POST" });
export const pauseSession = (id: string) =>
  apiFetch(`/simulation/sessions/${id}/pause`, { method: "POST" });
export const resumeSession = (id: string, ticks = 10) =>
  apiFetch(`/simulation/sessions/${id}/resume?ticks=${ticks}`, { method: "POST" });
export const stopSession = (id: string) =>
  apiFetch(`/simulation/sessions/${id}/stop`, { method: "POST" });

// Map
export const getTree = (session_id: string) =>
  apiFetch(`/map/tree?session_id=${session_id}`);
export const getPosts = (session_id: string, ptype?: string) =>
  apiFetch(`/map/posts?session_id=${session_id}${ptype ? `&ptype=${ptype}` : ""}`);
export const getPostContext = (post_id: string) =>
  apiFetch(`/map/posts/${post_id}/context`);

// Agents
export const getAgents = () => apiFetch("/agents");

// Analytics
export const getRecommendationStats = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/recommendation_stats`);
export const getSkipRateTimeline = (session_id: string, window = 10, step?: number) =>
  apiFetch(`/analytics/sessions/${session_id}/skip_rate_timeline?window=${window}${step ? `&step=${step}` : ""}`);
export const getUserModelTree = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/user_model_tree`);
export const getInferenceAccuracy = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/inference_accuracy`);
export const trainUserModel = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/train_user_model`, { method: "POST" });
export const getSessionProfiles = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/profiles`);
export const getTreeGrowth = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/tree_growth`);
export const getEvents = (session_id: string, limit = 50) =>
  apiFetch(`/analytics/sessions/${session_id}/events?limit=${limit}`);
export const getAgentStats = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/agent_stats`);
export const getActivityPerTick = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/activity`);

// User models / topic tree
export const getTopicTree = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/topic_tree`);
export const getUserModels = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/user_models`);
export const rebuildTopicTree = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/rebuild_topic_tree`, { method: "POST" });

// Persona priors + research metrics
export const computePriors = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/compute_priors`, { method: "POST" });
export const getPriors = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/priors`);
export const scoreQuality = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/score_quality`, { method: "POST" });
export const getResearchMetrics = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/research_metrics`);

// Persona topic headers (Feature 1)
export const generatePersonaTopics = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/generate_persona_topics`, { method: "POST" });
export const getPersonaTopics = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/persona_topics`);
export const getPersonaUserModels = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/persona_user_models`);
export const refinePersonaTopics = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/refine_persona_topics`, { method: "POST" });
export const scorePersonaActions = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/score_persona_actions`, { method: "POST" });
export const splitPersonaTopic = (session_id: string, topic_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/split_persona_topic`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ topic_id }),
  });
export const createUserFromDescription = (session_id: string, name: string, description: string) =>
  apiFetch(`/analytics/sessions/${session_id}/users_from_description`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, description }),
  });

// Per-agent action documents (Feature 2)
export const getAgentReports = (session_id: string) =>
  apiFetch(`/analytics/sessions/${session_id}/agent_reports`);
