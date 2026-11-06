import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type React from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import {
  AuthError,
  getStats,
  getTask,
  listInstances,
  listModels,
  listWorkflows,
  listTasks,
  shutdownApp,
  type AdminStats,
  type ExternalModelRecord,
  type InstanceRecord,
  type TaskDetailResponse,
  type TaskFilters,
  type TaskListResponse,
  type TaskRecord,
  type TaskStatus,
  type WorkflowListResponse
} from "./lib/api";
import { clearAdminToken, getAdminToken, setAdminToken } from "./lib/auth";
import { compactId } from "./lib/format";
import { connectAdminSocket, type AdminSocket } from "./lib/websocket";
import { AppShell } from "./components/app-shell";
import type { DashboardView } from "./components/app-shell";
import { ConfirmHost, confirmDialog } from "./components/confirm";
import { MediaThumb } from "./components/media-thumb";
import { InstancesPanel, ModelsPanel } from "./components/backend-panels";
import { PlaygroundPanel, type PlaygroundPrefill } from "./components/playground-panel";
import { SettingsPanel } from "./components/settings-panel";
import { StatusCards } from "./components/status-cards";
import { TaskFiltersBar } from "./components/task-filters";
import { TaskPreviewDrawer } from "./components/task-preview-drawer";
import { TaskTable } from "./components/task-table";
import { ToastHost } from "./components/toast";
import { TokenGate } from "./components/token-gate";
import type { ThemeMode } from "./components/theme-toggle";

const zeroCounts: Record<TaskStatus, number> = {
  pending: 0,
  queued: 0,
  running: 0,
  completed: 0,
  failed: 0
};

const TASK_PAGE_SIZE = 50;
const FILTER_KEYS = ["start", "end", "q", "status", "kind", "platform"] as const;

const emptyList: TaskListResponse = {
  total: 0,
  counts: zeroCounts,
  items: []
};

function readFiltersFromUrl(): TaskFilters {
  const params = new URLSearchParams(window.location.search);
  const filters: TaskFilters = {};
  for (const key of FILTER_KEYS) {
    const value = params.get(key);
    if (value) filters[key] = value;
  }
  return filters;
}

function writeFiltersToUrl(filters: TaskFilters): void {
  const params = new URLSearchParams();
  for (const key of FILTER_KEYS) {
    const value = filters[key];
    if (value) params.set(key, value);
  }
  const query = params.toString();
  window.history.replaceState(null, "", query ? `?${query}` : window.location.pathname);
}

export function App(): React.ReactElement {
  const [theme, setTheme] = useState<ThemeMode>(
    () =>
      (window.localStorage.getItem("comfyui2api.theme") as ThemeMode | null) ||
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")
  );
  const [filters, setFilters] = useState<TaskFilters>(readFiltersFromUrl);
  const [tasks, setTasks] = useState<TaskListResponse>(emptyList);
  const [taskPage, setTaskPage] = useState(0);
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [authNeeded, setAuthNeeded] = useState(() => !getAdminToken());
  const [live, setLive] = useState(false);
  const [selectedTask, setSelectedTask] = useState<TaskRecord | null>(null);
  const [detail, setDetail] = useState<TaskDetailResponse | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [activeView, setActiveView] = useState<DashboardView>("tasks");
  const [workflows, setWorkflows] = useState<WorkflowListResponse | null>(null);
  const [workflowsLoading, setWorkflowsLoading] = useState(false);
  const [instances, setInstances] = useState<InstanceRecord[]>([]);
  const [models, setModels] = useState<ExternalModelRecord[]>([]);
  const [backendLoading, setBackendLoading] = useState(false);
  const [quitting, setQuitting] = useState(false);
  const [playgroundPrefill, setPlaygroundPrefill] = useState<PlaygroundPrefill | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("comfyui2api.theme", theme);
  }, [theme]);

  // Filters are the single source of truth and apply immediately; the URL
  // mirrors them so a filtered view survives a reload and can be shared.
  const handleFiltersChange = useCallback((next: TaskFilters): void => {
    setFilters(next);
    setTaskPage(0);
    writeFiltersToUrl(next);
  }, []);

  const resetFilters = useCallback((): void => {
    setFilters({});
    setTaskPage(0);
    writeFiltersToUrl({});
  }, []);

  const apiFilters = useMemo(() => toApiFilters(filters), [filters]);

  // Refs mirror the latest values for the WebSocket handler below, so the
  // socket is not torn down and reconnected every time a filter or page
  // changes while still merging events against current state.
  const apiFiltersRef = useRef(apiFilters);
  apiFiltersRef.current = apiFilters;
  const taskPageRef = useRef(taskPage);
  taskPageRef.current = taskPage;

  const refresh = useCallback(async () => {
    if (!getAdminToken()) {
      setAuthNeeded(true);
      return;
    }
    setLoading(true);
    setError("");
    try {
      const [taskList, nextStats] = await Promise.all([
        listTasks(apiFilters, TASK_PAGE_SIZE, taskPage * TASK_PAGE_SIZE),
        getStats()
      ]);
      setTasks(taskList);
      setStats(nextStats);
      setAuthNeeded(false);
    } catch (err) {
      if (err instanceof AuthError) {
        clearAdminToken();
        setAuthNeeded(true);
      } else {
        setError(err instanceof Error ? err.message : "加载失败");
      }
    } finally {
      setLoading(false);
    }
  }, [apiFilters, taskPage]);

  useEffect(() => {
    if (authNeeded) return;
    void refresh();
  }, [authNeeded, refresh]);

  useEffect(() => {
    if (authNeeded) return;
    let closed = false;
    let retryId = 0;
    let socket: AdminSocket | null = null;

    function handleWsEvent(event: { type: string; data?: TaskListResponse; job?: TaskRecord; event?: string }): void {
      if (event.type === "snapshot" && event.data) {
        // Keep current page items; only refresh counts/total so pagination is
        // not clobbered by the socket's initial 50-item snapshot.
        setTasks((current) => ({
          total: event.data?.total ?? current.total,
          counts: event.data?.counts ?? current.counts,
          items: current.items
        }));
      }
      if (event.type === "task_updated" && event.job) {
        setTasks((current) => mergeTask(current, event.job as TaskRecord, event.event ?? "", taskPageRef.current, apiFiltersRef.current));
      }
    }

    // Reconnect on every close (throttled), so the socket survives repeated
    // drops instead of dying permanently after the first retry. The `closed`
    // flag stops the chain when the effect unmounts.
    const connect = (): void => {
      socket = connectAdminSocket({
        token: getAdminToken(),
        onOpen: () => setLive(true),
        onClose: () => {
          setLive(false);
          if (!closed) {
            retryId = window.setTimeout(connect, 2000);
          }
        },
        onEvent: handleWsEvent
      });
    };
    connect();

    return () => {
      closed = true;
      window.clearTimeout(retryId);
      socket?.close();
    };
  }, [authNeeded]);

  useEffect(() => {
    if (authNeeded || live) return;
    const id = window.setInterval(() => {
      void refresh();
    }, 2000);
    return () => window.clearInterval(id);
  }, [authNeeded, live, refresh]);

  const refreshBackend = useCallback(async () => {
    if (!getAdminToken()) return;
    setBackendLoading(true);
    try {
      const [instanceList, modelList, workflowList] = await Promise.all([listInstances(), listModels(), listWorkflows()]);
      setInstances(instanceList.items);
      setModels(modelList.items);
      setWorkflows(workflowList);
      setAuthNeeded(false);
    } catch (err) {
      if (err instanceof AuthError) {
        clearAdminToken();
        setAuthNeeded(true);
      } else {
        setError(err instanceof Error ? err.message : "后端配置加载失败");
      }
    } finally {
      setBackendLoading(false);
    }
  }, []);

  useEffect(() => {
    if (authNeeded || (activeView !== "instances" && activeView !== "models")) return;
    void refreshBackend();
    const id = window.setInterval(() => {
      void refreshBackend();
    }, 5000);
    return () => window.clearInterval(id);
  }, [activeView, authNeeded, refreshBackend]);

  useEffect(() => {
    if (!selectedTask) {
      setDetail(null);
      return;
    }
    setDetailLoading(true);
    getTask(selectedTask.job_id)
      .then((payload) => {
        setDetail(payload);
        setAuthNeeded(false);
      })
      .catch((err: unknown) => {
        if (err instanceof AuthError) {
          clearAdminToken();
          setAuthNeeded(true);
        } else {
          setError(err instanceof Error ? err.message : "详情加载失败");
        }
      })
      .finally(() => setDetailLoading(false));
  }, [selectedTask]);

  const refreshWorkflows = useCallback(async () => {
    setWorkflowsLoading(true);
    setError("");
    try {
      setWorkflows(await listWorkflows());
      setAuthNeeded(false);
    } catch (err) {
      if (err instanceof AuthError) {
        clearAdminToken();
        setAuthNeeded(true);
      } else {
        setError(err instanceof Error ? err.message : "工作流加载失败");
      }
    } finally {
      setWorkflowsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (authNeeded || activeView !== "workflows") return;
    void refreshWorkflows();
  }, [activeView, authNeeded, refreshWorkflows]);

  const handleShutdown = useCallback(async () => {
    if (quitting) return;
    const ok = await confirmDialog({
      title: "退出 comfyui2api",
      message: "当前服务会停止，所有进行中的任务将被中断。确定退出吗？",
      confirmLabel: "退出",
      danger: true
    });
    if (!ok) return;
    setQuitting(true);
    setError("");
    try {
      await shutdownApp();
      setError("comfyui2api 正在退出，窗口可以关闭。");
    } catch (err) {
      setQuitting(false);
      if (err instanceof AuthError) {
        clearAdminToken();
        setAuthNeeded(true);
      } else {
        setError(err instanceof Error ? err.message : "退出失败");
      }
    }
  }, [quitting]);

  const filterTasksByStatus = useCallback(
    (status: string): void => {
      handleFiltersChange(status ? { status } : {});
      setActiveView("tasks");
    },
    [handleFiltersChange]
  );

  // Drawer target from the playground: only the id is known there, the drawer
  // effect fetches the full record right after.
  const openTaskById = useCallback((jobId: string): void => {
    setSelectedTask({
      job_id: jobId,
      created_at: 0,
      created_at_utc: "",
      platform: "",
      kind: "",
      workflow: "",
      status: "queued",
      progress_percent: 0,
      output_count: 0
    });
  }, []);

  const reuseInPlayground = useCallback((prefill: PlaygroundPrefill): void => {
    setPlaygroundPrefill(prefill);
    setSelectedTask(null);
    setActiveView("playground");
  }, []);

  if (authNeeded) {
    return (
      <TokenGate
        onSubmit={(token) => {
          setAdminToken(token);
          setAuthNeeded(false);
          void refresh();
        }}
      />
    );
  }

  const counts = stats?.counts ?? tasks.counts ?? zeroCounts;
  const total = tasks.total || Object.values(counts).reduce((sum, value) => sum + value, 0);
  const viewMeta = viewTitles[activeView];

  return (
    <AppShell
      stats={stats}
      theme={theme}
      live={live}
      loading={loading}
      quitting={quitting}
      activeView={activeView}
      title={viewMeta.title}
      subtitle={viewMeta.subtitle}
      onNavigate={setActiveView}
      onThemeToggle={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
      onRefresh={() => void refresh()}
      onSettings={() => setSettingsOpen(true)}
      onShutdown={() => void handleShutdown()}
    >
      {error ? <div className="error-banner">{error}</div> : null}
      {activeView === "overview" ? (
        <div className="view-stack">
          <StatusCards counts={counts} total={total} healthyInstances={stats?.healthy_instance_count} onFilter={filterTasksByStatus} />
          <OverviewPanel stats={stats} total={total} />
        </div>
      ) : null}
      {activeView === "tasks" ? (
        <div className="view-stack">
          <StatusCards counts={counts} total={total} healthyInstances={stats?.healthy_instance_count} onFilter={filterTasksByStatus} />
          <TaskFiltersBar filters={filters} onChange={handleFiltersChange} onReset={resetFilters} />
          <TaskTable
            items={tasks.items}
            total={tasks.total}
            pageIndex={taskPage}
            pageSize={TASK_PAGE_SIZE}
            onPageChange={setTaskPage}
            onOpenTask={setSelectedTask}
          />
        </div>
      ) : null}
      {activeView === "playground" ? (
        <div className="view-stack">
          <PlaygroundPanel
            prefill={playgroundPrefill}
            onPrefillConsumed={() => setPlaygroundPrefill(null)}
            onOpenTask={openTaskById}
          />
        </div>
      ) : null}
      {activeView === "workflows" ? (
        <div className="view-stack">
          <WorkflowPanel workflows={workflows} loading={workflowsLoading} onRefresh={() => void refreshWorkflows()} />
        </div>
      ) : null}
      {activeView === "outputs" ? (
        <div className="view-stack">
          <OutputsPanel onOpenTask={setSelectedTask} />
        </div>
      ) : null}
      {activeView === "instances" ? (
        <div className="view-stack">
          <InstancesPanel items={instances} loading={backendLoading} onRefresh={() => void refreshBackend()} />
        </div>
      ) : null}
      {activeView === "models" ? (
        <div className="view-stack">
          <ModelsPanel
            items={models}
            instances={instances}
            workflows={workflows}
            loading={backendLoading}
            onRefresh={() => void refreshBackend()}
          />
        </div>
      ) : null}
      <TaskPreviewDrawer
        task={selectedTask}
        detail={detail}
        loading={detailLoading}
        onClose={() => setSelectedTask(null)}
        onReuseInPlayground={reuseInPlayground}
      />
      {settingsOpen ? (
        <SettingsPanel
          stats={stats}
          shuttingDown={quitting}
          onClose={() => setSettingsOpen(false)}
          onShutdown={() => void handleShutdown()}
        />
      ) : null}
      <ToastHost />
      <ConfirmHost />
    </AppShell>
  );
}

const viewTitles: Record<DashboardView, { title: string; subtitle: string }> = {
  overview: { title: "概览", subtitle: "运行状态、任务总览和本地目录" },
  tasks: { title: "任务记录", subtitle: "队列、状态、耗时、输出预览与失败原因" },
  playground: { title: "试运行", subtitle: "用公开 OpenAI 兼容接口验证文生图 / 图生图 / 文生视频 / 图生视频" },
  workflows: { title: "工作流", subtitle: "已加载工作流、类型和加载状态" },
  outputs: { title: "输出文件", subtitle: "最近任务产物和预览入口" },
  instances: { title: "实例", subtitle: "登记 ComfyUI 后端、健康状态和并发槽" },
  models: { title: "对外模型", subtitle: "工作流绑定、实例池和路由策略" }
};

function OverviewPanel({ stats, total }: { stats: AdminStats | null; total: number }): React.ReactElement {
  return (
    <section className="view-panel">
      <div className="panel-header">
        <div>
          <h2>本地运行概况</h2>
          <p>当前任务总数 {total}</p>
        </div>
      </div>
      <div className="runtime-grid">
        <RuntimeTile label="健康实例" value={`${stats?.healthy_instance_count ?? 0}/${stats?.instance_count ?? 0}`} />
        <RuntimeTile label="工作流目录" value={stats?.workflows_dir} />
        <RuntimeTile label="输出目录" value={stats?.runs_dir} />
        <RuntimeTile label="任务数据库" value={stats?.database_path} />
      </div>
    </section>
  );
}

function WorkflowPanel({
  workflows,
  loading,
  onRefresh
}: {
  workflows: WorkflowListResponse | null;
  loading: boolean;
  onRefresh: () => void;
}): React.ReactElement {
  const items = workflows?.items ?? [];
  return (
    <section className="view-panel">
      <div className="panel-header">
        <div>
          <h2>工作流列表</h2>
          <p>{workflows?.workflows_dir ?? "工作流目录"}</p>
        </div>
        <button className="ghost-button" type="button" onClick={onRefresh} disabled={loading}>
          刷新
        </button>
      </div>
      <div className="workflow-list">
        {items.map((item) => (
          <div className={item.available ? "workflow-row" : "workflow-row is-error"} key={item.name}>
            <div>
              <strong>{item.name}</strong>
              <span>{item.kind ?? "unknown"}</span>
            </div>
            <small>{item.available ? "可用" : item.load_error ?? item.parameter_error ?? "加载失败"}</small>
          </div>
        ))}
        {!loading && items.length === 0 ? <div className="empty-state compact">没有已加载的工作流</div> : null}
        {loading ? <div className="empty-state compact">正在加载工作流</div> : null}
      </div>
    </section>
  );
}

function OutputsPanel({ onOpenTask }: { onOpenTask: (task: TaskRecord) => void }): React.ReactElement {
  const [outputs, setOutputs] = useState<TaskRecord[]>([]);
  const [outputTotal, setOutputTotal] = useState(0);
  const [outputPage, setOutputPage] = useState(0);
  const [outputLoading, setOutputLoading] = useState(false);
  const [outputError, setOutputError] = useState("");
  const pageSize = 30;

  const loadOutputs = useCallback(async (page: number) => {
    if (!getAdminToken()) {
      setOutputError("未授权");
      return;
    }
    setOutputLoading(true);
    setOutputError("");
    try {
      // Server-side `has_outputs` filter keeps total and pagination truthful.
      const payload = await listTasks({ has_outputs: "1" }, pageSize, page * pageSize);
      setOutputs(payload.items);
      setOutputTotal(payload.total);
    } catch (err) {
      if (err instanceof AuthError) {
        clearAdminToken();
        setOutputError("凭证已失效，请重新登录");
      } else {
        setOutputError(err instanceof Error ? err.message : "输出加载失败");
      }
    } finally {
      setOutputLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadOutputs(outputPage);
  }, [loadOutputs, outputPage]);

  const pageCount = outputTotal > 0 ? Math.ceil(outputTotal / pageSize) : 0;

  return (
    <section className="view-panel">
      <div className="panel-header">
        <div>
          <h2>最近输出</h2>
          <p>有产物的任务，缩略图点击查看任务详情</p>
        </div>
      </div>
      {outputError ? <div className="error-banner">{outputError}</div> : null}
      <div className="output-grid">
        {outputLoading ? <div className="empty-state compact">正在加载输出…</div> : null}
        {!outputLoading &&
          outputs.map((task) => <OutputCard task={task} key={task.job_id} onOpen={() => onOpenTask(task)} />)}
        {!outputLoading && outputs.length === 0 ? <div className="empty-state compact">没有可显示的输出文件</div> : null}
      </div>
      <div className="pagination">
        <span>
          显示 {outputs.length} / 共 {outputTotal} 条带产物记录
          {pageCount > 0 ? ` · 第 ${outputPage + 1}/${pageCount} 页` : ""}
        </span>
        <div>
          <button type="button" disabled={outputPage === 0} onClick={() => setOutputPage(outputPage - 1)} title="上一页">
            <ChevronLeft size={16} />
          </button>
          <strong>{outputPage + 1}</strong>
          <button
            type="button"
            disabled={outputPage >= pageCount - 1}
            onClick={() => setOutputPage(outputPage + 1)}
            title="下一页"
          >
            <ChevronRight size={16} />
          </button>
        </div>
      </div>
    </section>
  );
}

function OutputCard({ task, onOpen }: { task: TaskRecord; onOpen: () => void }): React.ReactElement {
  return (
    <button className="output-card" type="button" onClick={onOpen} title={task.job_id}>
      <OutputThumb task={task} />
      <strong>{task.model_slug || task.workflow}</strong>
      <span>{compactId(task.job_id, 16)}</span>
      <small>{task.output_count} 个文件</small>
    </button>
  );
}

function OutputThumb({ task }: { task: TaskRecord }): React.ReactElement {
  return (
    <MediaThumb
      className="output-thumb-media"
      url={task.url || ""}
      mediaType={task.media_type}
      fallback={`${task.output_count} 个文件`}
    />
  );
}

function RuntimeTile({ label, value }: { label: string; value?: string }): React.ReactElement {
  return (
    <div className="runtime-card">
      <span>{label}</span>
      <strong>{value ?? "--"}</strong>
    </div>
  );
}

function toApiFilters(filters: TaskFilters): TaskFilters {
  return {
    ...filters,
    start: filters.start ? new Date(filters.start).toISOString() : undefined,
    end: filters.end ? new Date(filters.end).toISOString() : undefined
  };
}

function mergeTask(
  current: TaskListResponse,
  task: TaskRecord,
  eventName: string,
  pageIndex: number,
  filters: TaskFilters
): TaskListResponse {
  // Replace in-place if the job is already on this page. A brand-new task
  // (job_created) joins the top of page 0 so a submission shows up without a
  // manual refresh — but only when it actually matches the active filters,
  // otherwise the injected row and the inflated total would contradict the
  // server's next snapshot. Text/time filters cannot be checked cheaply, so
  // with q/start/end set the list is left to the next refresh.
  const exists = current.items.some((item) => item.job_id === task.job_id);
  if (!exists) {
    if (eventName !== "job_created") return current;
    const anyFreeText = Boolean(filters.q || filters.start || filters.end);
    const matches =
      !anyFreeText &&
      (["status", "kind", "platform"] as const).every((key) => {
        const raw = filters[key];
        if (!raw) return true;
        const allowed = raw.split(",").map((item) => item.trim()).filter(Boolean);
        return allowed.length === 0 || allowed.includes(String(task[key] ?? ""));
      });
    if (!matches) return current;
    const items = pageIndex === 0 ? [task, ...current.items].slice(0, TASK_PAGE_SIZE) : current.items;
    return {
      ...current,
      items,
      total: current.total + 1
    };
  }
  return {
    ...current,
    items: current.items.map((item) => (item.job_id === task.job_id ? task : item))
  };
}
