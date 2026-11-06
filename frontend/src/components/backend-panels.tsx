import type React from "react";
import { useState } from "react";
import {
  createInstance,
  createModel,
  deleteInstance,
  deleteModel,
  patchInstance,
  patchModel,
  type ExternalModelRecord,
  type InstanceRecord,
  type WorkflowListResponse
} from "../lib/api";
import { confirmDialog } from "./confirm";
import { toast } from "./toast";

export function InstancesPanel({
  items,
  loading,
  onRefresh
}: {
  items: InstanceRecord[];
  loading: boolean;
  onRefresh: () => void;
}): React.ReactElement {
  const [form, setForm] = useState({
    slug: "",
    display_name: "",
    base_url: "http://127.0.0.1:8188",
    auth_token: "",
    max_in_flight: "1",
    health_interval_s: ""
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [editing, setEditing] = useState<Record<string, {
    base_url: string;
    max_in_flight: string;
    health_interval_s: string;
    auth_token: string;
  }>>({});

  async function handleCreate(): Promise<void> {
    setBusy(true);
    setError("");
    try {
      await createInstance({
        slug: form.slug.trim(),
        display_name: form.display_name.trim() || null,
        base_url: form.base_url.trim(),
        auth_token: form.auth_token.trim() || null,
        max_in_flight: Number(form.max_in_flight || 1),
        health_interval_s: form.health_interval_s ? Number(form.health_interval_s) : null
      });
      setForm({ ...form, slug: "", display_name: "", auth_token: "" });
      onRefresh();
      toast(`实例 ${form.slug.trim()} 已添加`, "ok");
    } catch (err) {
      setError(err instanceof Error ? err.message : "创建失败");
    } finally {
      setBusy(false);
    }
  }

  function startEdit(item: InstanceRecord): void {
    setEditing((current) => ({
      ...current,
      [item.slug]: {
        base_url: item.base_url,
        max_in_flight: String(item.max_in_flight),
        health_interval_s: item.health_interval_s ? String(item.health_interval_s) : "",
        auth_token: ""
      }
    }));
  }

  function setEditField(slug: string, patch: Partial<{ base_url: string; max_in_flight: string; health_interval_s: string; auth_token: string }>): void {
    setEditing((current) => {
      const existing = current[slug];
      if (!existing) return current;
      return { ...current, [slug]: { ...existing, ...patch } };
    });
  }

  async function handleEditSave(slug: string): Promise<void> {
    const edited = editing[slug];
    if (!edited) return;
    setBusy(true);
    setError("");
    try {
      const body: Record<string, unknown> = {
        base_url: edited.base_url.trim(),
        max_in_flight: Number(edited.max_in_flight || 1)
      };
      if (edited.health_interval_s.trim()) {
        body.health_interval_s = Number(edited.health_interval_s);
      } else {
        body.health_interval_s = null;
      }
      if (edited.auth_token.trim()) {
        body.auth_token = edited.auth_token.trim();
      }
      await patchInstance(slug, body);
      setEditing((current) => {
        const next = { ...current };
        delete next[slug];
        return next;
      });
      onRefresh();
      toast(`实例 ${slug} 已保存`, "ok");
    } catch (err) {
      setError(err instanceof Error ? err.message : "更新失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="view-panel">
      <div className="panel-header">
        <div>
          <h2>ComfyUI 实例</h2>
          <p>添加后端后立刻探活；离线实例也会留在列表里。</p>
        </div>
        <button className="ghost-button" type="button" onClick={onRefresh} disabled={loading}>
          刷新
        </button>
      </div>
      {error ? <div className="error-banner">{error}</div> : null}
      <form
        className="backend-form"
        onSubmit={(event) => {
          event.preventDefault();
          void handleCreate();
        }}
      >
        <input placeholder="slug" value={form.slug} onChange={(event) => setForm({ ...form, slug: event.target.value })} required />
        <input placeholder="显示名" value={form.display_name} onChange={(event) => setForm({ ...form, display_name: event.target.value })} />
        <input placeholder="http://host:8188" value={form.base_url} onChange={(event) => setForm({ ...form, base_url: event.target.value })} required />
        <input placeholder="auth_token（可选）" value={form.auth_token} onChange={(event) => setForm({ ...form, auth_token: event.target.value })} />
        <input placeholder="max_in_flight" value={form.max_in_flight} onChange={(event) => setForm({ ...form, max_in_flight: event.target.value })} />
        <input placeholder="探活间隔秒（可空）" value={form.health_interval_s} onChange={(event) => setForm({ ...form, health_interval_s: event.target.value })} />
        <button className="primary-button" type="submit" disabled={busy}>
          添加实例
        </button>
      </form>
      <div className="workflow-list">
        {items.map((item) => {
          const isEditing = editing[item.slug];
          return (
            <div className="workflow-row" key={item.slug}>
              {isEditing ? (
                <div className="instance-edit-form">
                  <div>
                    <strong>{item.display_name || item.slug}</strong>
                    <span>编辑实例：URL / 并发 / 间隔 / 密钥</span>
                  </div>
                  <input
                    value={isEditing.base_url}
                    onChange={(event) => setEditField(item.slug, { base_url: event.target.value })}
                    aria-label="编辑 URL"
                    placeholder="http://host:8188"
                  />
                  <input
                    value={isEditing.max_in_flight}
                    onChange={(event) => setEditField(item.slug, { max_in_flight: event.target.value })}
                    aria-label="编辑最大并发"
                    placeholder="max_in_flight"
                  />
                  <input
                    value={isEditing.health_interval_s}
                    onChange={(event) => setEditField(item.slug, { health_interval_s: event.target.value })}
                    aria-label="编辑探活间隔"
                    placeholder="探活间隔秒（可空）"
                  />
                  <input
                    value={isEditing.auth_token}
                    type="password"
                    onChange={(event) => setEditField(item.slug, { auth_token: event.target.value })}
                    aria-label="编辑密钥"
                    placeholder="auth_token（留空不变）"
                  />
                  <div className="row-actions">
                    <button className="primary-button" type="button" disabled={busy} onClick={() => void handleEditSave(item.slug)}>
                      保存
                    </button>
                    <button
                      className="ghost-button"
                      type="button"
                      onClick={() =>
                        setEditing((current) => {
                          const next = { ...current };
                          delete next[item.slug];
                          return next;
                        })
                      }
                    >
                      取消
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <div>
                    <strong>{item.display_name || item.slug}</strong>
                    <span>
                      {item.base_url} · {item.health} · in_flight {item.in_flight}/{item.max_in_flight}
                    </span>
                    {item.last_error ? (
                      <span className="instance-error" title={item.last_error}>
                        {item.last_error}
                      </span>
                    ) : null}
                  </div>
                  <div className="row-actions">
                    <button className="ghost-button" type="button" onClick={() => startEdit(item)}>
                      编辑
                    </button>
                    <button
                      className="ghost-button"
                      type="button"
                      onClick={() =>
                        void patchInstance(item.slug, { enabled: !item.enabled })
                          .then(onRefresh)
                          .then(() => toast(`实例 ${item.slug} 已${item.enabled ? "停用" : "启用"}`, "ok"))
                          .catch((err: unknown) => setError(err instanceof Error ? err.message : "更新失败"))
                      }
                    >
                      {item.enabled ? "停用" : "启用"}
                    </button>
                    <button
                      className="ghost-button"
                      type="button"
                      onClick={() => {
                        void confirmDialog({
                          title: "删除实例",
                          message: `删除实例 ${item.slug}？将从 ${item.bound_model_count} 个模型解绑。`,
                          confirmLabel: "删除",
                          danger: true
                        }).then((ok) => {
                          if (!ok) return;
                          void deleteInstance(item.slug)
                            .then(onRefresh)
                            .then(() => toast(`实例 ${item.slug} 已删除`, "ok"))
                            .catch((err: unknown) => setError(err instanceof Error ? err.message : "删除失败"));
                        });
                      }}
                    >
                      删除
                    </button>
                  </div>
                </>
              )}
            </div>
          );
        })}
        {!loading && items.length === 0 ? <div className="empty-state compact">还没有实例，先添加一台 ComfyUI。</div> : null}
      </div>
    </section>
  );
}

export function ModelsPanel({
  items,
  instances,
  workflows,
  loading,
  onRefresh
}: {
  items: ExternalModelRecord[];
  instances: InstanceRecord[];
  workflows: WorkflowListResponse | null;
  loading: boolean;
  onRefresh: () => void;
}): React.ReactElement {
  const [form, setForm] = useState({
    slug: "",
    display_name: "",
    workflow_name: "",
    routing_policy: "round_robin",
    instance_slugs: [] as string[]
  });
  const [editing, setEditing] = useState<Record<string, {
    workflow_name: string;
    routing_policy: string;
    instance_slugs: string[];
  }>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const workflowItems = (workflows?.items ?? []).filter((item) => item.available);

  async function handleCreate(): Promise<void> {
    setBusy(true);
    setError("");
    try {
      await createModel({
        slug: form.slug.trim(),
        display_name: form.display_name.trim() || null,
        workflow_name: form.workflow_name || null,
        routing_policy: form.routing_policy,
        enabled: false,
        instance_slugs: form.instance_slugs
      });
      setForm({ ...form, slug: "", display_name: "" });
      onRefresh();
      toast(`模型 ${form.slug.trim()} 已创建（草稿，默认停用）`, "ok");
    } catch (err) {
      setError(err instanceof Error ? err.message : "创建失败");
    } finally {
      setBusy(false);
    }
  }

  function startEdit(item: ExternalModelRecord): void {
    setEditing((current) => ({
      ...current,
      [item.slug]: {
        workflow_name: item.workflow_name ?? "",
        routing_policy: item.routing_policy,
        instance_slugs: [...item.instance_slugs]
      }
    }));
  }

  function setEditField(slug: string, patch: Partial<{ workflow_name: string; routing_policy: string; instance_slugs: string[] }>): void {
    setEditing((current) => {
      const existing = current[slug];
      if (!existing) return current;
      return { ...current, [slug]: { ...existing, ...patch } };
    });
  }

  async function handleEditSave(slug: string): Promise<void> {
    const edited = editing[slug];
    if (!edited) return;
    setBusy(true);
    setError("");
    try {
      await patchModel(slug, {
        workflow_name: edited.workflow_name || null,
        routing_policy: edited.routing_policy,
        instance_slugs: edited.instance_slugs
      });
      setEditing((current) => {
        const next = { ...current };
        delete next[slug];
        return next;
      });
      onRefresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "更新失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="view-panel">
      <div className="panel-header">
        <div>
          <h2>对外模型</h2>
          <p>草稿默认停用。启用时必须能加载工作流文件。在行上点击「编辑」可配置模型绑定的实例。</p>
        </div>
        <button className="ghost-button" type="button" onClick={onRefresh} disabled={loading}>
          刷新
        </button>
      </div>
      {error ? <div className="error-banner">{error}</div> : null}
      <form
        className="backend-form"
        onSubmit={(event) => {
          event.preventDefault();
          void handleCreate();
        }}
      >
        <input placeholder="slug" value={form.slug} onChange={(event) => setForm({ ...form, slug: event.target.value })} required />
        <input placeholder="显示名" value={form.display_name} onChange={(event) => setForm({ ...form, display_name: event.target.value })} />
        <select value={form.workflow_name} onChange={(event) => setForm({ ...form, workflow_name: event.target.value })}>
          <option value="">选择工作流</option>
          {workflowItems.map((item) => (
            <option value={item.name} key={item.name}>
              {item.name}
            </option>
          ))}
        </select>
        <select value={form.routing_policy} onChange={(event) => setForm({ ...form, routing_policy: event.target.value })}>
          <option value="round_robin">轮询</option>
          <option value="random">随机</option>
        </select>
        <select
          multiple
          aria-label="关联实例"
          value={form.instance_slugs}
          onChange={(event) =>
            setForm({
              ...form,
              instance_slugs: Array.from(event.target.selectedOptions).map((option) => option.value)
            })
          }
        >
          {instances.map((item) => (
            <option value={item.slug} key={item.slug}>
              {item.display_name || item.slug}
            </option>
          ))}
        </select>
        <button className="primary-button" type="submit" disabled={busy}>
          创建草稿
        </button>
      </form>
      <div className="workflow-list">
        {items.map((item) => {
          const isEditing = editing[item.slug];
          return (
            <div className="workflow-row" key={item.slug}>
              {isEditing ? (
                <div className="model-edit-form">
                  <div>
                    <strong>{item.display_name || item.slug}</strong>
                    <span>编辑绑定：工作流 / 策略 / 关联实例</span>
                  </div>
                  <select
                    value={isEditing.workflow_name}
                    onChange={(event) => setEditField(item.slug, { workflow_name: event.target.value })}
                    aria-label="编辑工作流"
                  >
                    <option value="">选择工作流</option>
                    {workflowItems.map((wf) => (
                      <option value={wf.name} key={wf.name}>
                        {wf.name}
                      </option>
                    ))}
                  </select>
                  <select
                    value={isEditing.routing_policy}
                    onChange={(event) => setEditField(item.slug, { routing_policy: event.target.value })}
                    aria-label="编辑路由策略"
                  >
                    <option value="round_robin">轮询</option>
                    <option value="random">随机</option>
                  </select>
                  <select
                    multiple
                    value={isEditing.instance_slugs}
                    onChange={(event) =>
                      setEditField(item.slug, {
                        instance_slugs: Array.from(event.target.selectedOptions).map((option) => option.value)
                      })
                    }
                    aria-label="编辑关联实例"
                  >
                    {instances.map((inst) => (
                      <option value={inst.slug} key={inst.slug}>
                        {inst.display_name || inst.slug}
                      </option>
                    ))}
                  </select>
                  <div className="row-actions">
                    <button className="primary-button" type="button" disabled={busy} onClick={() => void handleEditSave(item.slug)}>
                      保存
                    </button>
                    <button
                      className="ghost-button"
                      type="button"
                      onClick={() =>
                        setEditing((current) => {
                          const next = { ...current };
                          delete next[item.slug];
                          return next;
                        })
                      }
                    >
                      取消
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <div>
                    <strong>{item.display_name || item.slug}</strong>
                    <span>
                      {item.workflow_name ?? "未绑定工作流"} · {item.routing_policy} · 实例{" "}
                      {item.instance_slugs.length > 0 ? item.instance_slugs.join(", ") : "（未绑定）"} · {item.ready ? "ready" : "not ready"}
                    </span>
                  </div>
                  <div className="row-actions">
                    <button className="ghost-button" type="button" onClick={() => startEdit(item)}>
                      编辑
                    </button>
                    <button
                      className="ghost-button"
                      type="button"
                      onClick={() =>
                        void patchModel(item.slug, { enabled: !item.enabled })
                          .then(onRefresh)
                          .then(() => toast(`模型 ${item.slug} 已${item.enabled ? "停用" : "启用"}`, "ok"))
                          .catch((err: unknown) => setError(err instanceof Error ? err.message : "更新失败"))
                      }
                    >
                      {item.enabled ? "停用" : "启用"}
                    </button>
                    <button
                      className="ghost-button"
                      type="button"
                      onClick={() => {
                        void confirmDialog({
                          title: "删除模型",
                          message: `删除模型 ${item.slug}？`,
                          confirmLabel: "删除",
                          danger: true
                        }).then((ok) => {
                          if (!ok) return;
                          void deleteModel(item.slug)
                            .then(onRefresh)
                            .then(() => toast(`模型 ${item.slug} 已删除`, "ok"))
                            .catch((err: unknown) => setError(err instanceof Error ? err.message : "删除失败"));
                        });
                      }}
                    >
                      删除
                    </button>
                  </div>
                </>
              )}
            </div>
          );
        })}
        {!loading && items.length === 0 ? <div className="empty-state compact">还没有对外模型。</div> : null}
      </div>
    </section>
  );
}
