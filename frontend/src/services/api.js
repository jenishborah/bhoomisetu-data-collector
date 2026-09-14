const API_BASE_URL = "https://bhoomisetu-api-dmcx.onrender.com/api";

export class ApiError extends Error {
  constructor(message, status = 0, detail = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function buildUrl(path, params) {
  const query = new URLSearchParams();
  Object.entries(params || {}).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") query.set(key, String(value));
  });
  const suffix = query.toString();
  return `${API_BASE_URL}${path}${suffix ? `?${suffix}` : ""}`;
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(buildUrl(path, options.params), {
      method: options.method || "GET",
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...options.headers,
      },
      body: options.body ? JSON.stringify(options.body) : undefined,
      signal: options.signal,
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError("The BhoomiSetu API is unavailable. Confirm that the local backend is running on port 8000.");
  }
  const isJson = response.headers.get("content-type")?.includes("application/json");
  const payload = isJson ? await response.json() : null;
  if (!response.ok) {
    const detail = payload?.detail;
    const message = typeof detail === "string" ? detail : `Request failed (${response.status}).`;
    throw new ApiError(message, response.status, detail);
  }
  return payload;
}

export function getDashboardOverview() { return request("/dashboard/overview"); }
export function getRiskOverview() { return request("/dashboard/risk-overview"); }
export const getDashboardRiskOverview = getRiskOverview;

export function getProjects(options = {}) {
  const params = typeof options === "number" ? { limit: options, offset: arguments[1] || 0 } : options;
  return request("/projects", { params });
}

export function getProject(projectId) { return request(`/projects/${encodeURIComponent(projectId)}`); }
export function getProjectRisk(projectId) { return request(`/projects/${encodeURIComponent(projectId)}/risk`); }
export function getProjectSnapshots(projectId) { return request(`/projects/${encodeURIComponent(projectId)}/snapshots`); }
export function predictProject(projectId) { return request(`/projects/${encodeURIComponent(projectId)}/predict`, { method: "POST" }); }
export function getEvidence(projectId) { return request(`/projects/${encodeURIComponent(projectId)}/evidence`); }
export function getActions(projectId) { return request(`/projects/${encodeURIComponent(projectId)}/actions`); }
export function simulateProject(projectId, payload) {
  return request(`/projects/${encodeURIComponent(projectId)}/simulate`, { method: "POST", body: payload });
}

export { API_BASE_URL };
