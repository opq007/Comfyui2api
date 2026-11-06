import { useEffect, useState } from "react";
import type React from "react";
import { AlertTriangle } from "lucide-react";

export interface ConfirmOptions {
  title: string;
  message: string;
  confirmLabel?: string;
  danger?: boolean;
}

interface PendingRequest extends ConfirmOptions {
  resolve: (ok: boolean) => void;
}

type Listener = (request: PendingRequest | null) => void;

let listener: Listener | null = null;

/** Promise-based in-app confirmation, replacing native window.confirm which
 * clashes with the dashboard's look and cannot be styled. */
export function confirmDialog(options: ConfirmOptions): Promise<boolean> {
  return new Promise((resolve) => {
    if (listener) {
      listener({ ...options, resolve });
    } else {
      resolve(false);
    }
  });
}

export function ConfirmHost(): React.ReactElement | null {
  const [request, setRequest] = useState<PendingRequest | null>(null);

  useEffect(() => {
    listener = setRequest;
    return () => {
      listener = null;
    };
  }, []);

  useEffect(() => {
    if (!request) return;
    const onKey = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        event.stopPropagation();
        request.resolve(false);
        setRequest(null);
      }
      if (event.key === "Enter") {
        request.resolve(true);
        setRequest(null);
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [request]);

  if (!request) return null;
  const settle = (ok: boolean): void => {
    request.resolve(ok);
    setRequest(null);
  };
  return (
    <div className="drawer-layer" role="alertdialog" aria-modal="true" aria-label={request.title}>
      <button className="drawer-scrim" type="button" onClick={() => settle(false)} aria-label="取消" />
      <aside className="confirm-card">
        {request.danger ? (
          <div className="confirm-icon danger">
            <AlertTriangle size={20} />
          </div>
        ) : null}
        <h3>{request.title}</h3>
        <p>{request.message}</p>
        <div className="row-actions">
          <button className="ghost-button" type="button" onClick={() => settle(false)}>
            取消
          </button>
          <button className={request.danger ? "danger-button" : "primary-button"} type="button" onClick={() => settle(true)}>
            {request.confirmLabel ?? "确定"}
          </button>
        </div>
      </aside>
    </div>
  );
}
