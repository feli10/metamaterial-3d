import { useEffect, useState } from "react";
import type { PropertyValue } from "../api/types";
import { fmtValue, parseValue } from "../lib/format";

/**
 * NumberField — a single property input (number | null, blank = auto).
 *
 * Keeps a local text buffer while typing (so partial input like "0." isn't fought) and only
 * COMMITS the parsed value on blur. When the value changes externally (Generate fills it, or
 * Load), it resyncs the box — but not while focused, so it never clobbers what you're typing.
 *
 * Shared by the matrix cells and the volume-fraction box so they behave identically.
 */

interface Props {
  value: PropertyValue;
  onCommit: (v: PropertyValue) => void;
  className?: string;
  placeholder?: string;
}

export default function NumberField({
  value,
  onCommit,
  className,
  placeholder = "auto",
}: Props) {
  const [text, setText] = useState(fmtValue(value));
  const [focused, setFocused] = useState(false);

  useEffect(() => {
    if (!focused) setText(fmtValue(value));
  }, [value, focused]);

  return (
    <input
      className={className}
      inputMode="decimal"
      placeholder={placeholder}
      value={text}
      onFocus={() => setFocused(true)}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => {
        setFocused(false);
        onCommit(parseValue(text));
      }}
    />
  );
}
