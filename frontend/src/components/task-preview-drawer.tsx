import type React from "react";
import { useEffect, useState } from "react";
import { Copy, Download, ExternalLink, FileQuestion, FlaskConical, LoaderCircle, X } from "lucide-react";
import type { TaskDetailResponse, TaskOutput, TaskRecord } from "../lib/api";
import { asJson, compactId, formatDateTime, formatDuration } from "../lib/format";
import { MediaThumb } from "./media-thumb";
import { kindLabels, StatusBadge } from "./status-badge";
import { copyText } from "./toast";
import { buildPrefillFromTask, taskNegativeText, taskPromptText } from "./playground-panel";
import type { PlaygroundPrefill } from "./playground-panel";

interface TaskPreviewDrawerProps {
  task: TaskRecord | null;
  detail: TaskDetailResponse | null;
  loading: boolean;
  onClose: () => void;
  onReuseInPlayground?: (prefill: PlaygroundPrefill) => void;
}

export function TaskPreviewDrawer({
  task,
  detail,
  loading,
  onClose,
  onReuseInPlayground
}: TaskPreviewDrawerProps): React.ReactElement | null {
  const [activeOutputIdx, setActiveOutputIdx] = useState(0);

  // Reset the gallery selection whenever another task is inspected.
  useEffect(() => {
    setActiveOutputIdx(0);
  }, [task?.job_id, detail]);

  // Esc closes the drawer, matching the scrim click.
  useEffect(() => {
    if (!task) return;
    const onKey = (event: KeyboardEvent): void => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [task, onClose]);

  if (!task) return null;
  const current = detail?.task ?? task;
  const outputs = detail?.outputs ?? task.outputs ?? [];
  const requestJson = current.request_json ?? {};
  const standardParams = (requestJson.standard_params as Record<string, unknown> | undefined) ?? {};
  const promptFull = taskPromptText(current);
  const negativeFull = taskNegativeText(current);
  const activeOutput = outputs[activeOutputIdx] ?? outputs[0];

  return (
    <div className="drawer-layer" role="dialog" aria-modal="true">
      <button className="drawer-scrim" type="button" onClick={onClose} aria-label="关闭" />
      <aside className="task-drawer">
        <div className="drawer-title">
          <div>
            <h2>任务详情</h2>
            <span title={current.job_id}>{compactId(current.job_id, 34)}</span>
          </div>
          <button className="icon-button" type="button" onClick={onClose} title="关闭 (Esc)">
            <X size={18} />
          </button>
        </div>

        {loading ? (
          <div className="drawer-loading">
            <LoaderCircle size={20} />
            加载中
          </div>
        ) : null}

        <div className="preview-stage">
          {activeOutput ? <TaskMedia output={activeOutput} /> : <EmptyPreview />}
        </div>
        {outputs.length > 1 ? (
          <div className="output-thumb-strip">
            {outputs.map((output, index) => (
              <button
                type="button"
                key={`${output.filename}-${index}`}
                className={index === activeOutputIdx ? "active" : ""}
                onClick={() => setActiveOutputIdx(index)}
                title={output.filename}
              >
                <MediaThumb url={output.url} mediaType={output.media_type} fallback={<FileQuestion size={16} />} />
              </button>
            ))}
          </div>
        ) : null}

        <div className="detail-grid">
          <Info label="状态" value={<StatusBadge status={current.status} />} />
          <Info label="耗时" value={formatDuration(current.duration_s)} />
          <Info label="平台" value={current.platform} />
          <Info label="类型" value={kindLabels[current.kind] ?? current.kind} />
          <Info label="模型" value={current.model_slug ?? current.requested_model ?? "--"} />
          <Info label="实例" value={current.instance_slug ?? "排队中"} />
          <Info label="prompt_id" value={current.prompt_id ? compactId(current.prompt_id, 18) : "--"} />
          <Info label="queue_number" value={String(current.queue_number ?? "--")} />
          <Info label="提交时间" value={formatDateTime(current.created_at_utc)} />
          <Info label="结束时间" value={formatDateTime(current.finished_at_utc)} />
        </div>

        {current.error ? (
          <section className="drawer-section error-block">
            <div className="section-title">
              <h3>错误信息</h3>
              <button type="button" onClick={() => void copyText(current.error ?? "", "错误信息已复制")}>
                <Copy size={14} />
                复制
              </button>
            </div>
            <pre>{current.error}</pre>
          </section>
        ) : null}

        {promptFull ? (
          <section className="drawer-section">
            <div className="section-title">
              <h3>提示词</h3>
              <button type="button" onClick={() => void copyText(promptFull, "提示词已复制")}>
                <Copy size={14} />
                复制
              </button>
            </div>
            <p className="prompt-block">{promptFull}</p>
            {negativeFull ? (
              <>
                <div className="section-title">
                  <h3>反向提示词</h3>
                  <button type="button" onClick={() => void copyText(negativeFull, "反向提示词已复制")}>
                    <Copy size={14} />
                    复制
                  </button>
                </div>
                <p className="prompt-block muted">{negativeFull}</p>
              </>
            ) : null}
          </section>
        ) : null}

        <section className="drawer-section">
          <h3>生效参数</h3>
          <div className="param-chips">
            {paramEntries(current, standardParams).map(([key, value]) => (
              <span className="param-chip" key={key}>
                <em>{key}</em>
                <strong title={String(value)}>{String(value)}</strong>
              </span>
            ))}
            {paramEntries(current, standardParams).length === 0 ? <span className="muted">无标准参数</span> : null}
          </div>
          <details className="raw-json">
            <summary>原始请求 JSON</summary>
            <pre>{asJson(requestJson)}</pre>
          </details>
        </section>

        <section className="drawer-section output-list">
          <h3>输出文件</h3>
          {outputs.length > 0 ? (
            outputs.map((output) => (
              <div className="output-file-row" key={`${output.filename}-${output.url}`}>
                <a href={output.url} target="_blank" rel="noreferrer">
                  <span>{output.filename}</span>
                  <ExternalLink size={14} />
                </a>
                <a
                  className="icon-button"
                  href={output.url}
                  download={output.filename}
                  title="下载文件"
                  onClick={(event) => event.stopPropagation()}
                >
                  <Download size={14} />
                </a>
              </div>
            ))
          ) : (
            <span className="muted">暂无输出</span>
          )}
        </section>

        <div className="drawer-actions">
          {onReuseInPlayground ? (
            <button className="primary-button" type="button" onClick={() => onReuseInPlayground(buildPrefillFromTask(current))}>
              <FlaskConical size={15} />
              在试运行中重跑
            </button>
          ) : null}
          <button type="button" onClick={() => void copyText(current.url ?? "", "输出地址已复制")} disabled={!current.url}>
            <Copy size={15} />
            复制 URL
          </button>
          <a className={current.url ? "ghost-button" : "ghost-button disabled"} href={current.url ?? "#"} target="_blank" rel="noreferrer">
            <ExternalLink size={15} />
            打开输出文件
          </a>
        </div>
      </aside>
    </div>
  );
}

function paramEntries(task: TaskRecord, standardParams: Record<string, unknown>): Array<[string, unknown]> {
  const entries: Array<[string, unknown]> = [];
  const workflow = task.request_json?.workflow ?? task.workflow;
  if (workflow) entries.push(["workflow", workflow]);
  const size = task.request_json?.size;
  if (size) entries.push(["size", size]);
  for (const key of ["steps", "cfg", "seed", "n", "seconds", "duration", "fps", "frames", "quality"]) {
    const value = standardParams[key];
    if (value !== undefined && value !== null && typeof value !== "object") entries.push([key, value]);
  }
  return entries;
}

function Info({ label, value }: { label: string; value: React.ReactNode }): React.ReactElement {
  return (
    <div className="info-tile">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function TaskMedia({ output }: { output: TaskOutput }): React.ReactElement {
  const mediaType = output.media_type ?? "";
  if (mediaType.startsWith("image/")) {
    return <img src={output.url} alt={output.filename} />;
  }
  if (mediaType.startsWith("video/")) {
    return <video src={output.url} controls />;
  }
  if (mediaType.startsWith("audio/")) {
    return <audio src={output.url} controls />;
  }
  return (
    <a className="file-preview" href={output.url} target="_blank" rel="noreferrer">
      <FileQuestion size={28} />
      <span>{output.filename}</span>
    </a>
  );
}

function EmptyPreview(): React.ReactElement {
  return (
    <div className="empty-preview">
      <FileQuestion size={30} />
      <span>暂无预览</span>
    </div>
  );
}
