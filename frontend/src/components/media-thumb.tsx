import type React from "react";

interface MediaThumbProps {
  url: string;
  mediaType?: string | null;
  className?: string;
  /** Rendered when the media cannot be previewed (unknown type / no URL). */
  fallback?: React.ReactNode;
}

/** Single implementation of the small preview thumbnails used by the task
 * table, outputs grid, playground history and output gallery strips. */
export function MediaThumb({ url, mediaType, className, fallback = null }: MediaThumbProps): React.ReactElement {
  if (url && mediaType?.startsWith("image/")) {
    return <img className={className} src={url} alt="" loading="lazy" />;
  }
  if (url && mediaType?.startsWith("video/")) {
    return <video className={className} src={url} muted playsInline preload="metadata" />;
  }
  return <span className={`${className ?? ""} media-thumb-fallback`.trim()}>{fallback}</span>;
}
