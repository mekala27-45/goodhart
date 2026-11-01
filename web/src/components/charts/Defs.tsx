// Fill patterns, defined once per chart so an id means the same texture everywhere. Texture carries
// the difference as well as hue: a series survives greyscale printing and colour vision deficiency.
export const PATTERN = {
  /** The gap between what the reward paid and what the twins confirmed: a diagonal in the reward colour. */
  gap: "pat-gap",
  /** The floor: the control gray, hatched. */
  floor: "pat-floor",
  /** A registered window: the predicted steps, a light hatch. */
  window: "pat-window",
};

export function ChartDefs({ uid }: { uid: string }) {
  return (
    <defs>
      <pattern id={`${PATTERN.gap}-${uid}`} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="6" height="6" fill="var(--gh-reward)" opacity="0.08" />
        <line x1="0" y1="0" x2="0" y2="6" stroke="var(--gh-reward)" strokeWidth="1.3" opacity="0.55" />
      </pattern>
      <pattern id={`${PATTERN.floor}-${uid}`} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="6" height="6" fill="var(--gh-control)" opacity="0.2" />
        <line x1="0" y1="0" x2="0" y2="6" stroke="var(--gh-control)" strokeWidth="1.6" opacity="0.8" />
      </pattern>
      <pattern id={`${PATTERN.window}-${uid}`} width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)">
        <rect width="8" height="8" fill="var(--gh-seq-1)" opacity="0.28" />
        <line x1="0" y1="0" x2="0" y2="8" stroke="var(--gh-seq-2)" strokeWidth="1" opacity="0.45" />
      </pattern>
    </defs>
  );
}

export const fill = (pattern: string, uid: string) => `url(#${pattern}-${uid})`;
