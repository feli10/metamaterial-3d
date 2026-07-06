import type { PropertyValue } from "../api/types";

/**
 * Shared conversions between a stored PropertyValue (number | null) and the text a user
 * types. Used by every property input (matrix cells + volume fraction) so "blank = auto"
 * behaves identically everywhere.
 */

/** Stored value -> text for display. null (auto) shows as an empty box. */
export function fmtValue(v: PropertyValue): string {
  return v === null ? "" : String(v);
}

/** Typed text -> stored value. Empty or non-numeric => null (auto). */
export function parseValue(text: string): PropertyValue {
  const t = text.trim();
  if (t === "") return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}
