import type React from "react";
import { useEffect, useMemo, useState } from "react";
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
  type VisibilityState
} from "@tanstack/react-table";
import { ChevronLeft, ChevronRight, Columns3, Copy, Eye, FileQuestion } from "lucide-react";
import type { TaskRecord } from "../lib/api";
import { compactId, formatDateTime, formatDuration } from "../lib/format";
import { MediaThumb } from "./media-thumb";
import { ProgressCell } from "./progress-cell";
import { kindLabels, StatusBadge } from "./status-badge";
import { copyText } from "./toast";

const COLUMN_STORAGE_KEY = "comfyui2api.tasks.columns";

interface TaskTableProps {
  items: TaskRecord[];
  total: number;
  pageIndex: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onOpenTask: (task: TaskRecord) => void;
}

function defaultColumnVisibility(): VisibilityState {
  return { finished_at_utc: false, platform: false };
}

function restoreColumnVisibility(): VisibilityState {
  try {
    const raw = window.localStorage.getItem(COLUMN_STORAGE_KEY);
    if (raw) {
      return { ...defaultColumnVisibility(), ...(JSON.parse(raw) as VisibilityState) };
    }
  } catch {
    // corrupted snapshot falls back to defaults
  }
  return defaultColumnVisibility();
}

function promptSnippet(task: TaskRecord): string {
  return task.prompt_full || task.prompt_preview || "";
}

export function TaskTable({ items, total, pageIndex, pageSize, onPageChange, onOpenTask }: TaskTableProps): React.ReactElement {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>(restoreColumnVisibility);

  useEffect(() => {
    window.localStorage.setItem(COLUMN_STORAGE_KEY, JSON.stringify(columnVisibility));
  }, [columnVisibility]);

  const columns = useMemo<ColumnDef<TaskRecord>[]>(
    () => [
      {
        id: "thumb",
        header: "",
        enableSorting: false,
        cell: ({ row }) => <ThumbCell task={row.original} />
      },
      {
        accessorKey: "created_at_utc",
        header: "提交时间",
        cell: ({ row }) => (
          <span className="nowrap" title={row.original.created_at_utc}>
            {formatDateTime(row.original.created_at_utc, true)}
          </span>
        )
      },
      {
        accessorKey: "finished_at_utc",
        header: "结束时间",
        cell: ({ row }) => <span title={row.original.finished_at_utc ?? ""}>{formatDateTime(row.original.finished_at_utc, true)}</span>
      },
      {
        accessorKey: "duration_s",
        header: "耗时",
        cell: ({ row }) => <span className="duration-pill">{formatDuration(row.original.duration_s)}</span>
      },
      {
        accessorKey: "platform",
        header: "平台",
        cell: ({ row }) => <span className={`platform-pill platform-${row.original.platform.toLowerCase().replace(/[^a-z]/g, "")}`}>{row.original.platform}</span>
      },
      {
        accessorKey: "kind",
        header: "类型",
        cell: ({ row }) => <span className="kind-pill">{kindLabels[row.original.kind] ?? row.original.kind}</span>
      },
      {
        id: "prompt",
        header: "提示词",
        enableSorting: false,
        cell: ({ row }) => {
          const text = promptSnippet(row.original);
          if (!text) return <span className="muted">--</span>;
          return (
            <span className="prompt-snippet" title={text}>
              {text}
            </span>
          );
        }
      },
      {
        accessorKey: "job_id",
        header: "任务 ID",
        cell: ({ row }) => (
          <button
            className="id-button"
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              void copyText(row.original.job_id, "任务 ID 已复制");
            }}
            title={row.original.job_id}
          >
            <span>{compactId(row.original.job_id, 14)}</span>
            <Copy size={14} />
          </button>
        )
      },
      {
        accessorKey: "status",
        header: "任务状态",
        cell: ({ row }) => <StatusBadge status={row.original.status} />
      },
      {
        accessorKey: "progress_percent",
        header: "进度",
        cell: ({ row }) => <ProgressCell value={row.original.progress_percent} status={row.original.status} />
      },
      {
        id: "detail",
        header: "详情",
        cell: ({ row }) => (
          <button
            className="detail-button"
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              onOpenTask(row.original);
            }}
          >
            <Eye size={15} />
            预览
          </button>
        )
      }
    ],
    [onOpenTask]
  );

  const table = useReactTable({
    data: items,
    columns,
    state: {
      sorting,
      columnVisibility
    },
    onSortingChange: setSorting,
    onColumnVisibilityChange: setColumnVisibility,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel()
  });

  const rows = table.getRowModel().rows;
  const pageCount = total > 0 ? Math.ceil(total / pageSize) : 0;
  const canPrev = pageIndex > 0;
  const canNext = pageIndex < pageCount - 1;

  return (
    <section className="table-panel">
      <div className="panel-header">
        <div>
          <h2>任务队列与历史记录</h2>
          <p>WebSocket 实时推送，断线后自动轮询；排序仅对当前页生效</p>
        </div>
        <details className="column-menu">
          <summary>
            <Columns3 size={16} />
            列设置
          </summary>
          <div>
            {table.getAllLeafColumns().map((column) => (
              <label key={column.id}>
                <input
                  type="checkbox"
                  checked={column.getIsVisible()}
                  onChange={column.getToggleVisibilityHandler()}
                />
                {String(column.columnDef.header ?? column.id)}
              </label>
            ))}
          </div>
        </details>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th key={header.id} className={header.column.id === "thumb" || header.column.id === "prompt" ? "no-sort" : ""}>
                    {header.isPlaceholder ? null : (
                      <button
                        type="button"
                        onClick={header.column.getToggleSortingHandler()}
                        disabled={header.column.id === "thumb" || header.column.id === "prompt"}
                      >
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        <span>{header.column.getIsSorted() === "asc" ? "↑" : header.column.getIsSorted() === "desc" ? "↓" : ""}</span>
                      </button>
                    )}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id} className="clickable" onClick={() => onOpenTask(row.original)}>
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
                ))}
              </tr>
            ))}
            {rows.length === 0 ? (
              <tr>
                <td className="empty-state" colSpan={columns.length}>
                  没有匹配的任务
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
      <div className="pagination">
        <span>
          显示 {items.length} / 共 {total} 条记录{pageCount > 0 ? ` · 第 ${pageIndex + 1}/${pageCount} 页` : ""}
        </span>
        <div>
          <button type="button" onClick={() => onPageChange(pageIndex - 1)} disabled={!canPrev} title="上一页">
            <ChevronLeft size={16} />
          </button>
          <strong>{pageCount > 0 ? pageIndex + 1 : 1}</strong>
          <button type="button" onClick={() => onPageChange(pageIndex + 1)} disabled={!canNext} title="下一页">
            <ChevronRight size={16} />
          </button>
        </div>
      </div>
    </section>
  );
}

function ThumbCell({ task }: { task: TaskRecord }): React.ReactElement {
  const url = task.outputs?.[0]?.url || task.url || "";
  const mediaType = task.outputs?.[0]?.media_type || task.media_type || "";
  return (
    <MediaThumb
      className="row-thumb"
      url={url}
      mediaType={mediaType}
      fallback={task.output_count > 0 ? <FileQuestion size={16} /> : null}
    />
  );
}
