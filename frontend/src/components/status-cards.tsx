import type React from "react";
import { Activity, AlertCircle, CheckCircle2, CircleDot, ListChecks } from "lucide-react";
import type { TaskStatus } from "../lib/api";

interface StatusCardsProps {
  counts: Record<TaskStatus, number>;
  total: number;
  healthyInstances?: number;
  /** Clicking a card jumps to the task list pre-filtered on that status. */
  onFilter?: (status: string) => void;
}

export function StatusCards({ counts, total, healthyInstances = 0, onFilter }: StatusCardsProps): React.ReactElement {
  const successRate = total > 0 ? Math.round((counts.completed / total) * 1000) / 10 : 0;
  const cards: Array<{
    label: string;
    value: number;
    hint: string;
    icon: typeof ListChecks;
    tone: string;
    status?: string;
  }> = [
    { label: "总任务", value: total, hint: "历史记录", icon: ListChecks, tone: "green" },
    { label: "运行中", value: counts.running, hint: `${healthyInstances} 健康实例`, icon: Activity, tone: "blue", status: "running" },
    { label: "排队中", value: counts.queued + counts.pending, hint: "等待空槽", icon: CircleDot, tone: "amber", status: "pending,queued" },
    { label: "成功", value: counts.completed, hint: `成功率 ${successRate}%`, icon: CheckCircle2, tone: "green", status: "completed" },
    { label: "失败", value: counts.failed, hint: "点击查看失败原因", icon: AlertCircle, tone: "red", status: "failed" }
  ];

  return (
    <section className="status-cards" aria-label="任务统计">
      {cards.map((card) => {
        const clickable = Boolean(card.status && onFilter);
        const content = (
          <>
            <div className={`stat-dot ${card.tone}`}>
              <card.icon size={14} />
            </div>
            <span>{card.label}</span>
            <strong>{card.value}</strong>
            <small>{card.hint}</small>
          </>
        );
        if (!clickable) {
          return (
            <div className="stat-card" key={card.label}>
              {content}
            </div>
          );
        }
        return (
          <button className="stat-card clickable" type="button" key={card.label} onClick={() => onFilter?.(card.status ?? "")}>
            {content}
          </button>
        );
      })}
    </section>
  );
}
