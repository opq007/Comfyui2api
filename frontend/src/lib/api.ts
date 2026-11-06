import { getAdminToken, getApiToken, clearApiToken } from "./auth";

export type TaskStatus = "pending" | "queued" | "running" | "completed" | "failed";

export interface TaskOutput {
  filename: string;
  url: string;
  media_type?: string | null;
  node_id?: string | null;
  output_key?: string | null;
}

export interface TaskRecord {
  job_id: string;
  created_at: number;
  created_at_utc: string;
  started_at_utc?: string | null;
  finished_at_utc?: string | null;
  updated_at_utc?: string | null;
  duration_s?: number | null;
  platform: string;
  kind: string;
  workflow: string;
  requested_model?: string | null;
  model_slug?: string | null;
  instance_slug?: string | null;
  status: TaskStatus;
  progress_percent: number;
  progress?: Record<string, unknown> | null;
  prompt_id?: string | null;
  queue_number?: number | null;
  current_node?: string | null;
  url?: string | null;
  output_count: number;
  error?: string | null;
  prompt_preview?: string | null;
  request_json?: Record<string, unknown>;
  outputs?: TaskOutput[];
}

export interface TaskListResponse {
  total: number;
  counts: Record<TaskStatus, number>;
  items: TaskRecord[];
}

export interface TaskDetailResponse {
  task: TaskRecord;
  outputs: TaskOutput[];
}

export interface AdminStats {
  counts: Record<TaskStatus, number>;
  instance_count: number;
  healthy_instance_count: number;
  workflows_dir: string;
  runs_dir: string;
  database_path: string;
  ui_enabled: boolean;
}

export type InstanceHealth = "unknown" | "healthy" | "unhealthy" | "disabled";

export interface InstanceRecord {
  slug: string;
  display_name?: string | null;
  base_url: string;
  enabled: boolean;
  max_in_flight: number;
  health_interval_s?: number | null;
  has_auth_token: boolean;
  health: InstanceHealth;
  consecutive_failures: number;
  last_check_at?: number | null;
  last_error?: string | null;
  in_flight: number;
  bound_model_count: number;
}

export interface ExternalModelRecord {
  slug: string;
  display_name?: string | null;
  workflow_name?: string | null;
  routing_policy: "round_robin" | "random";
  enabled: boolean;
  instance_slugs: string[];
  kind: string[];
  workflow_available: boolean;
  ready: boolean;
}

export interface TaskFilters {
  start?: string;
  end?: string;
  q?: string;
  status?: string;
  kind?: string;
  platform?: string;
}

export interface WorkflowItem {
  name: string;
  kind?: string | null;
  available: boolean;
  load_error?: string | null;
  parameter_error?: string | null;
}

export interface WorkflowListResponse {
  workflows_dir: string;
  items: WorkflowItem[];
}

export interface SnapshotEvent {
  type: "snapshot";
  data: TaskListResponse;
}

export interface TaskUpdatedEvent {
  type: "task_updated";
  event: string;
  ts: string;
  job: TaskRecord;
}

export type AdminWsEvent = SnapshotEvent | TaskUpdatedEvent;

export class AuthError extends Error {
  constructor() {
    super("Unauthorized");
  }
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

export class PublicAuthError extends Error {
  constructor() {
    super("Unauthorized");
  }
}

export type PlaygroundKind = "txt2img" | "img2img" | "txt2video" | "img2video" | "txt2audio";

export interface PublicModel {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  display_name?: string | null;
  kind: string[];
  ready: boolean;
  workflow_available: boolean;
}

export interface PublicJob {
  job_id: string;
  status: TaskStatus;
  kind?: string;
  progress_percent?: number;
  current_node?: string | null;
  error?: string | null;
  url?: string | null;
  outputs?: TaskOutput[];
  model_slug?: string | null;
  requested_model?: string | null;
  instance_slug?: string | null;
}

export interface PublicVideo {
  id: string;
  object: string;
  model?: string;
  status: string;
  progress: number;
  url?: string | null;
  error?: { message?: string } | null;
}

// Events pushed on the per-job WebSocket `GET /v1/jobs/{job_id}/ws`. The
// initial snapshot carries the full `PublicJob`; subsequent events carry
// deltas (progress, node, status transitions, final outputs).
export interface PublicJobSnapshotEvent {
  type: "job_snapshot";
  data: PublicJob;
}
export interface PublicJobProgressEvent {
  type: "job_progress";
  data: Record<string, unknown>;
}
export interface PublicJobRunningEvent {
  type: "job_running";
  data: { node?: string | null; instance_slug?: string | null };
}
export interface PublicJobQueuedEvent {
  type: "job_queued";
  data: { client_id?: string; workflow?: string };
}
export interface PublicJobCompletedEvent {
  type: "job_completed";
  data: { url?: string | null; outputs?: TaskOutput[] };
}
export interface PublicJobFailedEvent {
  type: "job_failed";
  data: { error?: string };
}
export interface PublicJobWsErrorEvent {
  type: "error";
  data: { message?: string };
}
export type PublicJobWsEvent =
  | PublicJobSnapshotEvent
  | PublicJobProgressEvent
  | PublicJobRunningEvent
  | PublicJobQueuedEvent
  | PublicJobCompletedEvent
  | PublicJobFailedEvent
  | PublicJobWsErrorEvent;

export interface PlaygroundRequestTrace {
  method: string;
  path: string;
  headers: Record<string, string>;
  body: unknown;
}

export interface PlaygroundSubmitResult {
  jobId?: string;
  videoId?: string;
  response: unknown;
  request: PlaygroundRequestTrace;
}

export async function listTasks(filters: TaskFilters, limit = 200, offset = 0): Promise<TaskListResponse> {
  const params = new URLSearchParams();
  params.set("limit", String(Math.min(200, Math.max(1, limit))));
  params.set("offset", String(Math.max(0, offset)));
  for (const [key, value] of Object.entries(filters)) {
    if (value) params.set(key, value);
  }
  return requestJson<TaskListResponse>(`/v1/admin/tasks?${params.toString()}`);
}

export async function getTask(jobId: string): Promise<TaskDetailResponse> {
  return requestJson<TaskDetailResponse>(`/v1/admin/tasks/${encodeURIComponent(jobId)}`);
}

export async function getStats(): Promise<AdminStats> {
  return requestJson<AdminStats>("/v1/admin/stats");
}

export async function listWorkflows(): Promise<WorkflowListResponse> {
  return requestJson<WorkflowListResponse>("/v1/admin/workflows");
}

export async function listInstances(): Promise<{ items: InstanceRecord[] }> {
  return requestJson<{ items: InstanceRecord[] }>("/v1/admin/instances");
}

export async function createInstance(body: Record<string, unknown>): Promise<InstanceRecord> {
  return requestJson<InstanceRecord>("/v1/admin/instances", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
}

export async function patchInstance(slug: string, body: Record<string, unknown>): Promise<InstanceRecord> {
  return requestJson<InstanceRecord>(`/v1/admin/instances/${encodeURIComponent(slug)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
}

export async function deleteInstance(slug: string): Promise<{ status: string }> {
  return requestJson<{ status: string }>(`/v1/admin/instances/${encodeURIComponent(slug)}`, { method: "DELETE" });
}

export async function listModels(): Promise<{ items: ExternalModelRecord[] }> {
  return requestJson<{ items: ExternalModelRecord[] }>("/v1/admin/models");
}

export async function createModel(body: Record<string, unknown>): Promise<ExternalModelRecord> {
  return requestJson<ExternalModelRecord>("/v1/admin/models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
}

export async function patchModel(slug: string, body: Record<string, unknown>): Promise<ExternalModelRecord> {
  return requestJson<ExternalModelRecord>(`/v1/admin/models/${encodeURIComponent(slug)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
}

export async function deleteModel(slug: string): Promise<{ status: string }> {
  return requestJson<{ status: string }>(`/v1/admin/models/${encodeURIComponent(slug)}`, { method: "DELETE" });
}

export async function shutdownApp(): Promise<{ status: string }> {
  return requestJson<{ status: string }>("/v1/admin/shutdown", { method: "POST" });
}

export function adminWsUrl(token: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const url = new URL("/v1/admin/tasks/ws", `${protocol}//${window.location.host}`);
  if (token) url.searchParams.set("token", token);
  return url.toString();
}

/** Per-job public WS endpoint. Browsers cannot set WS headers, so the API
 * token is passed as `?token=` (the WS auth parser accepts both). */
export function publicJobWsUrl(jobId: string, token: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const url = new URL(`/v1/jobs/${encodeURIComponent(jobId)}/ws`, `${protocol}//${window.location.host}`);
  if (token) url.searchParams.set("token", token);
  return url.toString();
}

async function requestJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  const token = getAdminToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  const response = await fetch(path, { ...init, headers });
  if (response.status === 401) {
    throw new AuthError();
  }
  if (!response.ok) {
    throw new ApiError(await errorMessage(response), response.status);
  }
  return (await response.json()) as T;
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { error?: { message?: string }; detail?: unknown };
    return payload.error?.message ?? String(payload.detail ?? response.statusText);
  } catch {
    return response.statusText;
  }
}

function publicAuthHeader(): string {
  const token = getApiToken();
  return token ? `Bearer ${token}` : "";
}

async function requestPublicJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  const auth = publicAuthHeader();
  if (auth) headers.set("Authorization", auth);
  const response = await fetch(path, { ...init, headers });
  if (response.status === 401) {
    clearApiToken();
    throw new PublicAuthError();
  }
  if (!response.ok) {
    throw new ApiError(await errorMessage(response), response.status);
  }
  return (await response.json()) as T;
}

export async function listPublicModels(): Promise<PublicModel[]> {
  const payload = await requestPublicJson<{ data?: PublicModel[] }>("/v1/models");
  return Array.isArray(payload.data) ? payload.data : [];
}

export async function getPublicJob(jobId: string): Promise<PublicJob> {
  const payload = await requestPublicJson<{ job: PublicJob }>(`/v1/jobs/${encodeURIComponent(jobId)}`);
  return payload.job;
}

export async function getPublicVideo(videoId: string): Promise<PublicVideo> {
  return requestPublicJson<PublicVideo>(`/v1/videos/${encodeURIComponent(videoId)}`);
}

function omitEmpty(values: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(values)) {
    if (value === undefined || value === null) continue;
    if (typeof value === "string" && !value.trim()) continue;
    out[key] = typeof value === "string" ? value.trim() : value;
  }
  return out;
}

export interface PlaygroundFields {
  model: string;
  prompt: string;
  negative_prompt?: string;
  n?: string;
  size?: string;
  width?: string;
  height?: string;
  steps?: string;
  cfg?: string;
  seed?: string;
  response_format?: string;
  seconds?: string;
  duration?: string;
  fps?: string;
  frames?: string;
  quality?: string;
  metadata?: string;
  voice?: string;
  speed?: string;
  imageFile?: File | null;
  imageUrl?: string;
}

function imageStandardFields(fields: PlaygroundFields): Record<string, unknown> {
  return omitEmpty({
    model: fields.model,
    prompt: fields.prompt,
    negative_prompt: fields.negative_prompt,
    n: fields.n,
    size: fields.size,
    width: fields.width,
    height: fields.height,
    steps: fields.steps,
    cfg: fields.cfg,
    seed: fields.seed,
    response_format: fields.response_format
  });
}

function videoStandardFields(fields: PlaygroundFields): Record<string, unknown> {
  return omitEmpty({
    model: fields.model,
    prompt: fields.prompt,
    negative_prompt: fields.negative_prompt,
    size: fields.size,
    width: fields.width,
    height: fields.height,
    steps: fields.steps,
    cfg: fields.cfg,
    seed: fields.seed,
    seconds: fields.seconds || fields.duration,
    duration: fields.duration,
    fps: fields.fps,
    frames: fields.frames,
    quality: fields.quality,
    metadata: fields.metadata
  });
}

export async function submitPlayground(kind: PlaygroundKind, fields: PlaygroundFields): Promise<PlaygroundSubmitResult> {
  if (kind === "txt2img") {
    const body = imageStandardFields(fields);
    const request: PlaygroundRequestTrace = {
      method: "POST",
      path: "/v1/images/generations",
      headers: { Authorization: "Bearer ***", "Content-Type": "application/json", "x-comfyui-async": "1" },
      body
    };
    const response = await requestPublicJson<{ job_id?: string; status?: string }>("/v1/images/generations", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-comfyui-async": "1" },
      body: JSON.stringify(body)
    });
    return { jobId: response.job_id, response, request };
  }

  if (kind === "img2img") {
    const file = fields.imageFile ?? null;
    const imageUrl = (fields.imageUrl || "").trim();
    if (file) {
      const form = new FormData();
      for (const [key, value] of Object.entries(imageStandardFields(fields))) {
        form.append(key, String(value));
      }
      form.append("image", file, file.name);
      const request: PlaygroundRequestTrace = {
        method: "POST",
        path: "/v1/images/edits",
        headers: { Authorization: "Bearer ***", "x-comfyui-async": "1" },
        body: { ...imageStandardFields(fields), image: `<file:${file.name}>` }
      };
      const headers = new Headers({ Accept: "application/json", "x-comfyui-async": "1" });
      const auth = publicAuthHeader();
      if (auth) headers.set("Authorization", auth);
      const http = await fetch("/v1/images/edits", { method: "POST", headers, body: form });
      if (http.status === 401) {
        clearApiToken();
        throw new PublicAuthError();
      }
      if (!http.ok) throw new ApiError(await errorMessage(http), http.status);
      const response = (await http.json()) as { job_id?: string };
      return { jobId: response.job_id, response, request };
    }
    const body = { ...imageStandardFields(fields), image: imageUrl };
    const request: PlaygroundRequestTrace = {
      method: "POST",
      path: "/v1/images/edits",
      headers: { Authorization: "Bearer ***", "Content-Type": "application/json", "x-comfyui-async": "1" },
      body
    };
    const response = await requestPublicJson<{ job_id?: string }>("/v1/images/edits", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-comfyui-async": "1" },
      body: JSON.stringify(body)
    });
    return { jobId: response.job_id, response, request };
  }

  if (kind === "txt2audio") {
    const body = omitEmpty({
      model: fields.model,
      input: fields.prompt,
      voice: fields.voice,
      speed: fields.speed
    });
    const request: PlaygroundRequestTrace = {
      method: "POST",
      path: "/v1/audio/speech",
      headers: { Authorization: "Bearer ***", "Content-Type": "application/json", "x-comfyui-async": "1" },
      body
    };
    const response = await requestPublicJson<{ job_id?: string }>("/v1/audio/speech", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-comfyui-async": "1" },
      body: JSON.stringify(body)
    });
    return { jobId: response.job_id, response, request };
  }

  const file = kind === "img2video" ? fields.imageFile ?? null : null;
  const imageUrl = kind === "img2video" ? (fields.imageUrl || "").trim() : "";
  if (file) {
    const form = new FormData();
    for (const [key, value] of Object.entries(videoStandardFields(fields))) {
      form.append(key, String(value));
    }
    form.append("input_reference", file, file.name);
    const request: PlaygroundRequestTrace = {
      method: "POST",
      path: "/v1/videos",
      headers: { Authorization: "Bearer ***" },
      body: { ...videoStandardFields(fields), input_reference: `<file:${file.name}>` }
    };
    const headers = new Headers({ Accept: "application/json" });
    const auth = publicAuthHeader();
    if (auth) headers.set("Authorization", auth);
    const http = await fetch("/v1/videos", { method: "POST", headers, body: form });
    if (http.status === 401) {
      clearApiToken();
      throw new PublicAuthError();
    }
    if (!http.ok) throw new ApiError(await errorMessage(http), http.status);
    const response = (await http.json()) as { id?: string };
    return { videoId: response.id, response, request };
  }

  const body = videoStandardFields(fields);
  if (imageUrl) body.input_reference = imageUrl;
  const request: PlaygroundRequestTrace = {
    method: "POST",
    path: "/v1/videos",
    headers: { Authorization: "Bearer ***", "Content-Type": "application/json" },
    body
  };
  const response = await requestPublicJson<{ id?: string }>("/v1/videos", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  return { videoId: response.id, response, request };
}
