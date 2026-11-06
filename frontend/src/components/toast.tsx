import { useEffect, useState } from "react";
import type React from "react";
import { CheckCircle2, Info, XCircle } from "lucide-react";

export type ToastTone = "ok" | "error" | "info";

export interface ToastItem {
  id: number;
  message: string;
  tone: ToastTone;
}

type Listener = (item: ToastItem) => void;

let nextId = 1;
const listeners = new Set<Listener>();

/** Fire a transient toast from anywhere without threading a context through
 * the tree. The single <ToastHost /> mounted in App receives it. */
export function toast(message: string, tone: ToastTone = "ok"): void {
  const item: ToastItem = { id: nextId++, message, tone };
  for (const listener of listeners) listener(item);
}

/** Copy to the clipboard with toast feedback; the shared shape behind every
 * copy button on the dashboard. Returns false when there was nothing to copy. */
export async function copyText(text: string, message = "已复制"): Promise<boolean> {
  if (!text) return false;
  await navigator.clipboard.writeText(text);
  toast(message, "ok");
  return true;
}

const TONE_ICON: Record<ToastTone, typeof Info> = {
  ok: CheckCircle2,
  error: XCircle,
  info: Info
};

export function ToastHost(): React.ReactElement | null {
  const [items, setItems] = useState<ToastItem[]>([]);

  useEffect(() => {
    const listener: Listener = (item) => {
      setItems((current) => [...current, item]);
      window.setTimeout(() => {
        setItems((current) => current.filter((entry) => entry.id !== item.id));
      }, 3200);
    };
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }, []);

  if (items.length === 0) return null;
  return (
    <div className="toast-stack" aria-live="polite">
      {items.map((item) => {
        const Icon = TONE_ICON[item.tone];
        return (
          <div className={`toast toast-${item.tone}`} key={item.id}>
            <Icon size={15} />
            <span>{item.message}</span>
          </div>
        );
      })}
    </div>
  );
}
