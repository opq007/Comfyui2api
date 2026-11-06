import type React from "react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { ImagePlus, KeyRound, LoaderCircle, Play } from "lucide-react";
import {
  getPublicJob,
  listPublicModels,
  PublicAuthError,
  submitPlayground,
  type PlaygroundKind,
  type PlaygroundRequestTrace,
  type PublicJob,
  type PublicJobWsEvent,
  type PublicModel,
  type PublicVideo,
  type TaskOutput
} from "../lib/api";
import { connectPublicJobSocket, type PublicJobSocket } from "../lib/websocket";
import { clearApiToken, getApiToken, setApiToken } from "../lib/auth";
import { asJson } from "../lib/format";
import { kindLabels } from "./status-badge";

const modes: Array<{ kind: PlaygroundKind; label: string }> = [
  { kind: "txt2img", label: "文生图" },
  { kind: "img2img", label: "图生图" },
  { kind: "txt2video", label: "文生视频" },
  { kind: "img2video", label: "图生视频" },
  { kind: "txt2audio", label: "文生语音" }
];

const emptyFields = {
  model: "",
  prompt: "",
  negative_prompt: "",
  n: "",
  size: "",
  width: "",
  height: "",
  steps: "",
  cfg: "",
  seed: "",
  response_format: "url",
  seconds: "",
  duration: "",
  fps: "",
  frames: "",
  quality: "",
  metadata: "",
  voice: "",
  speed: "",
  imageUrl: ""
};

type FieldState = typeof emptyFields;

export function PlaygroundPanel(): React.ReactElement {
  const [apiTokenNeeded, setApiTokenNeeded] = useState(() => !getApiToken());
  const [tokenInput, setTokenInput] = useState("");
  const [kind, setKind] = useState<PlaygroundKind>("txt2img");
  const [fields, setFields] = useState<FieldState>(emptyFields);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [models, setModels] = useState<PublicModel[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [requestTrace, setRequestTrace] = useState<PlaygroundRequestTrace | null>(null);
  const [rawResponse, setRawResponse] = useState<unknown>(null);
  const [job, setJob] = useState<PublicJob | null>(null);
  const [video, setVideo] = useState<PublicVideo | null>(null);
  const [submittedMeta, setSubmittedMeta] = useState<{ modelName: string; modelSlug: string } | null>(null);
  const [showTrace, setShowTrace] = useState(false);

  const needsImage = kind === "img2img" || kind === "img2video";
  const isVideo = kind === "txt2video" || kind === "img2video";
  const isAudio = kind === "txt2audio";

  const filteredModels = useMemo(
    () => models.filter((item) => (item.kind || []).includes(kind)),
    [kind, models]
  );

  const selected = filteredModels.find((item) => item.id === fields.model) ?? null;
  const selectedReady = Boolean(selected?.ready);

  const setField = (key: keyof FieldState, value: string): void => {
    setFields((current) => ({ ...current, [key]: value }));
  };

  const loadModels = useCallback(async () => {
    if (!getApiToken()) {
      setApiTokenNeeded(true);
      return;
    }
    setModelsLoading(true);
    setError("");
    try {
      const items = await listPublicModels();
      setModels(items);
      setApiTokenNeeded(false);
    } catch (err) {
      if (err instanceof PublicAuthError) {
        setApiTokenNeeded(true);
        setModels([]);
      } else {
        setError(err instanceof Error ? err.message : "模型列表加载失败");
      }
    } finally {
      setModelsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (apiTokenNeeded) return;
    void loadModels();
  }, [apiTokenNeeded, loadModels]);

  useEffect(() => {
    if (!fields.model) return;
    if (filteredModels.some((item) => item.id === fields.model)) return;
    setFields((current) => ({ ...current, model: "" }));
  }, [fields.model, filteredModels]);

  // Live status for a submitted job/video. Use the push-based per-job WebSocket
  // (one event per actual state change) instead of polling HTTP every 1.5s for
  // the whole generation. If WS is unavailable/fails repeatedly, fall back to
  // a slower HTTP poll so the progress bar still advances.
  useEffect(() => {
    const jobId = job?.job_id;
    const isVideoLive = video && video.status !== "completed" && video.status !== "failed";
    if (!jobId || job.status === "completed" || job.status === "failed") return;

    const token = getApiToken();
    let cancelled = false;
    let socket: PublicJobSocket | null = null;
    let pollTimer = 0;
    let reconnectTimer = 0;
    let attempt = 0;
    let fallbackActive = false;

    const applyWsEvent = (entry: PublicJobWsEvent): void => {
      switch (entry.type) {
        case "job_snapshot":
          setJob(entry.data);
          break;
        case "job_completed": {
          // Delta: url + outputs on top of the existing snapshot.
          const url = entry.data.url ?? null;
          const outputs = entry.data.outputs ?? [];
          setJob((current) =>
            current ? { ...current, status: "completed", url, outputs } : current
          );
          if (isVideoLive && url) {
            setVideo((current) => (current ? { ...current, status: "completed", url } : current));
          }
          break;
        }
        case "job_progress": {
          const value = Number(entry.data.value ?? 0);
          const max = Number(entry.data.max ?? 0);
          const percent = max > 0 ? Math.max(0, Math.min(99, Math.round((value / max) * 100))) : 0;
          setJob((current) => (current ? { ...current, progress_percent: percent } : current));
          if (isVideoLive) setVideo((current) => (current ? { ...current, progress: percent } : current));
          break;
        }
        case "job_running":
          setJob((current) =>
            current
              ? {
                  ...current,
                  status: "running",
                  current_node: entry.data.node ?? null,
                  instance_slug: entry.data.instance_slug ?? current.instance_slug ?? null
                }
              : current
          );
          if (isVideoLive && video?.status !== "running") {
            setVideo((current) => (current ? { ...current, status: "in_progress" } : current));
          }
          break;
        case "job_queued":
          setJob((current) => (current ? { ...current, status: "queued" } : current));
          break;
        case "job_failed":
          setJob((current) => (current ? { ...current, status: "failed", error: entry.data.error ?? null } : current));
          if (isVideoLive) {
            setVideo((current) => (current ? { ...current, status: "failed", error: { message: entry.data.error ?? "" } } : current));
          }
          break;
        case "error":
          setError(entry.data.message ?? "任务查询失败");
          break;
        default:
          break;
      }
    };

    const startFallbackPolling = (): void => {
      if (fallbackActive) return;
      fallbackActive = true;
      const tick = async (): Promise<void> => {
        if (cancelled) return;
        try {
          // Video is stored under its job_id alias; reuse the same public job.
          const next = await getPublicJob(jobId);
          if (cancelled) return;
          setJob(next);
          if (isVideoLive && next.status === "completed" && next.url) {
            setVideo((current) => (current ? { ...current, status: "completed", url: next.url } : current));
          }
        } catch (err) {
          if (cancelled) return;
          if (err instanceof PublicAuthError) {
            setApiTokenNeeded(true);
            return;
          }
          setError(err instanceof Error ? err.message : "任务查询失败");
        }
      };
      void tick();
      pollTimer = window.setInterval(() => void tick(), 3000);
    };

    const open = (): void => {
      if (cancelled) return;
      socket = connectPublicJobSocket({
        jobId,
        token,
        onOpen: () => {
          attempt = 0;
        },
        onClose: () => {
          if (cancelled) return;
          socket = null;
          if (!fallbackActive) {
            const delay = Math.min(30000, 1500 * 2 ** attempt);
            attempt += 1;
            reconnectTimer = window.setTimeout(() => {
              if (!cancelled) open();
            }, delay);
            // Give up on WS after a few attempts and degrade to polling.
            if (attempt >= 5) startFallbackPolling();
          }
        },
        onEvent: applyWsEvent
      });
    };
    open();

    return () => {
      cancelled = true;
      window.clearTimeout(reconnectTimer);
      window.clearInterval(pollTimer);
      socket?.close();
    };
  }, [job?.job_id, job?.status, video?.status]);

  async function handleSubmit(event: React.FormEvent): Promise<void> {
    event.preventDefault();
    const prompt = fields.prompt.trim();
    if (!fields.model) {
      setError("请选择模型");
      return;
    }
    if (!prompt) {
      setError("请填写 prompt");
      return;
    }
    if (needsImage && !imageFile && !fields.imageUrl.trim()) {
      setError("请上传参考图或填写图片 URL");
      return;
    }
    if (!selectedReady) {
      setError("该模型未就绪，无法提交");
      return;
    }
    setSubmitting(true);
    setError("");
    setJob(null);
    setVideo(null);
    // Snapshot which model this submission targets so the result meta row stays
    // truthful even if the picker changes mid-run.
    setSubmittedMeta({ modelName: selected?.display_name || fields.model, modelSlug: fields.model });
    try {
      const result = await submitPlayground(kind, {
        ...fields,
        imageFile: needsImage ? imageFile : null,
        imageUrl: needsImage ? fields.imageUrl : ""
      });
      setRequestTrace(result.request);
      setRawResponse(result.response);
      if (result.jobId) {
        setJob({ job_id: result.jobId, status: "pending", progress_percent: 0 });
      }
      if (result.videoId) {
        setVideo({ id: result.videoId, object: "video", status: "queued", progress: 0 });
      }
    } catch (err) {
      setSubmittedMeta(null);
      if (err instanceof PublicAuthError) {
        setApiTokenNeeded(true);
      } else {
        setError(err instanceof Error ? err.message : "提交失败");
      }
    } finally {
      setSubmitting(false);
    }
  }

  if (apiTokenNeeded) {
    return (
      <section className="view-panel playground-panel">
        <div className="panel-header">
          <div>
            <h2>试运行需要 API_TOKEN</h2>
            <p>公开 /v1 接口不接受 ADMIN_TOKEN。关闭标签页后需要重新输入。</p>
          </div>
        </div>
        <form
          className="playground-token"
          onSubmit={(event) => {
            event.preventDefault();
            const token = tokenInput.trim();
            if (!token) return;
            setApiToken(token);
            setTokenInput("");
            setApiTokenNeeded(false);
          }}
        >
          <label>
            <KeyRound size={16} />
            <input
              type="password"
              value={tokenInput}
              onChange={(event) => setTokenInput(event.target.value)}
              placeholder="API_TOKEN"
              autoFocus
            />
          </label>
          <button className="primary-button" type="submit" disabled={!tokenInput.trim()}>
            保存并加载模型
          </button>
        </form>
      </section>
    );
  }

  const previewUrl = job?.url || job?.outputs?.[0]?.url || video?.url || "";
  const progress = isVideo ? video?.progress ?? 0 : job?.progress_percent ?? 0;
  const statusText = isVideo ? video?.status ?? "" : job?.status ?? "";
  const failedMessage = job?.error || video?.error?.message || "";
  const busy = submitting || statusText === "pending" || statusText === "queued" || statusText === "running" || statusText === "in_progress";

  return (
    <section className="view-panel playground-panel">
      <div className="panel-header">
        <div>
          <h2>公开接口试运行</h2>
          <p>走真实 OpenAI 兼容 /v1，任务会进入任务记录。</p>
        </div>
        <div className="row-actions">
          <button className="ghost-button" type="button" onClick={() => void loadModels()} disabled={modelsLoading}>
            刷新模型
          </button>
          <button
            className="ghost-button"
            type="button"
            onClick={() => {
              clearApiToken();
              setApiTokenNeeded(true);
              setModels([]);
            }}
          >
            更换 API_TOKEN
          </button>
        </div>
      </div>
      {error ? <div className="error-banner playground-error">{error}</div> : null}
      <div className="playground-split">
        <form className="playground-form" onSubmit={(event) => void handleSubmit(event)}>
          <div className="playground-modes" role="tablist" aria-label="生成模式">
            {modes.map((item) => (
              <button
                key={item.kind}
                className={kind === item.kind ? "active" : ""}
                type="button"
                role="tab"
                aria-selected={kind === item.kind}
                onClick={() => setKind(item.kind)}
              >
                {item.label}
              </button>
            ))}
          </div>
          <label className="playground-field span-all">
            <span>model</span>
            <select value={fields.model} onChange={(event) => setField("model", event.target.value)} required>
              <option value="">{modelsLoading ? "加载中…" : "选择公开模型"}</option>
              {filteredModels.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.display_name || item.id}
                  {item.ready ? "" : "（未就绪）"}
                </option>
              ))}
            </select>
            {!modelsLoading && models.length === 0 ? (
              <small>没有已启用的对外模型，请先到「对外模型」创建并启用。</small>
            ) : null}
            {!modelsLoading && models.length > 0 && filteredModels.length === 0 ? (
              <small>没有支持 {kindLabels[kind] ?? kind} 的公开模型。</small>
            ) : null}
            {selected && !selected.ready ? <small>工作流不可用或没有健康实例，无法提交。</small> : null}
          </label>
          <label className="playground-field span-all">
            <span>prompt</span>
            <textarea
              rows={3}
              value={fields.prompt}
              onChange={(event) => setField("prompt", event.target.value)}
              placeholder="a cute cat, pixel art"
              required
            />
          </label>
          <label className="playground-field span-all">
            <span>negative_prompt</span>
            <textarea
              rows={2}
              value={fields.negative_prompt}
              onChange={(event) => setField("negative_prompt", event.target.value)}
              placeholder="可选"
            />
          </label>
          {isAudio ? (
            <>
              <label className="playground-field">
                <span>voice</span>
                <input value={fields.voice} onChange={(event) => setField("voice", event.target.value)} placeholder="可选，如 alloy" />
              </label>
              <label className="playground-field">
                <span>speed</span>
                <input value={fields.speed} onChange={(event) => setField("speed", event.target.value)} placeholder="可选，如 1.0" />
              </label>
            </>
          ) : null}
          {needsImage ? (
            <div className="playground-field span-all">
              <span>参考图（文件优先，否则 URL）</span>
              <div className="playground-image">
                <label className="file-drop">
                  <ImagePlus size={14} />
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(event) => setImageFile(event.target.files?.[0] ?? null)}
                  />
                  <em>{imageFile ? imageFile.name : "选择本地图片"}</em>
                </label>
                <input
                  placeholder="https://… 图片 URL"
                  value={fields.imageUrl}
                  onChange={(event) => setField("imageUrl", event.target.value)}
                />
              </div>
            </div>
          ) : null}
          {!isAudio ? (
            <>
              <label className="playground-field">
                <span>size</span>
                <input value={fields.size} onChange={(event) => setField("size", event.target.value)} placeholder="1024x1024" />
              </label>
              <label className="playground-field">
                <span>width</span>
                <input value={fields.width} onChange={(event) => setField("width", event.target.value)} placeholder="可选" />
              </label>
              <label className="playground-field">
                <span>height</span>
                <input value={fields.height} onChange={(event) => setField("height", event.target.value)} placeholder="可选" />
              </label>
            </>
          ) : null}
          {!isVideo && !isAudio ? (
            <label className="playground-field">
              <span>n</span>
              <input value={fields.n} onChange={(event) => setField("n", event.target.value)} placeholder="1" />
            </label>
          ) : null}
          <label className="playground-field">
            <span>steps</span>
            <input value={fields.steps} onChange={(event) => setField("steps", event.target.value)} />
          </label>
          <label className="playground-field">
            <span>cfg</span>
            <input value={fields.cfg} onChange={(event) => setField("cfg", event.target.value)} />
          </label>
          <label className="playground-field">
            <span>seed</span>
            <input value={fields.seed} onChange={(event) => setField("seed", event.target.value)} />
          </label>
          {!isVideo ? (
            <label className="playground-field">
              <span>response_format</span>
              <select value={fields.response_format} onChange={(event) => setField("response_format", event.target.value)}>
                <option value="url">url</option>
                <option value="b64_json">b64_json</option>
              </select>
            </label>
          ) : (
            <>
              <label className="playground-field">
                <span>seconds</span>
                <input value={fields.seconds} onChange={(event) => setField("seconds", event.target.value)} placeholder="duration 别名" />
              </label>
              <label className="playground-field">
                <span>duration</span>
                <input value={fields.duration} onChange={(event) => setField("duration", event.target.value)} />
              </label>
              <label className="playground-field">
                <span>fps</span>
                <input value={fields.fps} onChange={(event) => setField("fps", event.target.value)} />
              </label>
              <label className="playground-field">
                <span>frames</span>
                <input value={fields.frames} onChange={(event) => setField("frames", event.target.value)} />
              </label>
              <label className="playground-field">
                <span>quality</span>
                <input value={fields.quality} onChange={(event) => setField("quality", event.target.value)} placeholder="standard" />
              </label>
              <label className="playground-field span-2">
                <span>metadata</span>
                <input value={fields.metadata} onChange={(event) => setField("metadata", event.target.value)} placeholder="可选 JSON 或字符串" />
              </label>
            </>
          )}
          <div className="playground-actions span-all">
            <button className="primary-button" type="submit" disabled={submitting || !selectedReady || !fields.model}>
              {submitting ? <LoaderCircle size={16} /> : <Play size={16} />}
              {submitting ? "提交中" : "生成"}
            </button>
            <small>{isAudio ? "POST /v1/audio/speech · async" : isVideo ? "POST /v1/videos" : kind === "img2img" ? "POST /v1/images/edits · async" : "POST /v1/images/generations · async"}</small>
          </div>
        </form>
        <div className="playground-result">
          {submittedMeta ? (
            <div className="playground-result-meta">
              <span>
                <em>模型</em>
                <strong title={submittedMeta.modelSlug}>{submittedMeta.modelName}</strong>
                <code>{submittedMeta.modelSlug}</code>
              </span>
              <span>
                <em>实例</em>
                <strong title={job?.instance_slug ?? ""}>{job?.instance_slug?.trim() || "等待分配…"}</strong>
              </span>
            </div>
          ) : null}
          <div className="preview-stage playground-preview">
            {previewUrl && (isAudio || looksAudio(previewUrl)) ? (
              <audio src={previewUrl} controls />
            ) : previewUrl && (isVideo || looksVideo(previewUrl)) ? (
              <video src={previewUrl} controls />
            ) : previewUrl ? (
              <img src={previewUrl} alt="生成结果" />
            ) : busy ? (
              <div className="empty-preview">
                <LoaderCircle size={22} />
                <span>{statusText || "排队"} · {progress}%</span>
              </div>
            ) : failedMessage ? (
              <div className="empty-preview">{failedMessage}</div>
            ) : (
              <div className="empty-preview">提交后在此预览</div>
            )}
          </div>
          {job || video ? (
            <div className="progress-cell playground-progress">
              <div className="progress-track">
                <div className={`progress-fill${busy ? " is-running" : ""}`} style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} />
              </div>
              <span>{progress}%</span>
            </div>
          ) : null}
          {failedMessage ? <div className="error-banner">{failedMessage}</div> : null}
          <button className="ghost-button" type="button" onClick={() => setShowTrace((value) => !value)}>
            {showTrace ? "收起请求对照" : "查看请求对照"}
          </button>
          {showTrace ? (
            <div className="playground-trace">
              <pre>{asJson(requestTrace)}</pre>
              <pre>{asJson(rawResponse)}</pre>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}

function looksVideo(url: string): boolean {
  return /\.(mp4|webm|mov|gif)(\?|$)/i.test(url);
}

function looksAudio(url: string): boolean {
  return /\.(wav|mp3|flac|ogg|m4a|aac|opus)(\?|$)/i.test(url);
}
