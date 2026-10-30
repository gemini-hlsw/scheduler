const DARK_COLOR = "#374151";
const LIT_COLOR = "#fde68a";
const OUTLINE_COLOR = "#9ca3af";

// The moon as seen from a site: the lit part drawn over a dark disk.
export default function MoonPhaseIcon({
  illumination,
  waxing,
  southern,
  size = 16,
}: {
  // Lit fraction, 0 (new) to 1 (full).
  illumination: number;
  waxing: boolean;
  // Seen from the southern hemisphere the moon is upside down, so a waxing
  // moon is lit on the left instead of the right.
  southern: boolean;
  size?: number;
}) {
  const c = size / 2;
  const r = c - 1;
  // The terminator is half an ellipse: as wide as the disk at new and full
  // moon, a straight line at the quarters.
  const terminatorRx = r * Math.abs(1 - 2 * illumination);
  // Back up the terminator through the lit side for a crescent, through the
  // dark side for a gibbous moon.
  const sweep = illumination < 0.5 ? 0 : 1;
  const litPath =
    `M ${c} ${c - r} A ${r} ${r} 0 0 1 ${c} ${c + r} ` +
    `A ${terminatorRx} ${r} 0 0 ${sweep} ${c} ${c - r} Z`;
  const litOnRight = waxing !== southern;

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      aria-label={`Moon ${Math.round(illumination * 100)}% illuminated`}
    >
      <circle cx={c} cy={c} r={r} fill={DARK_COLOR} />
      <path
        d={litPath}
        fill={LIT_COLOR}
        transform={litOnRight ? undefined : `matrix(-1 0 0 1 ${size} 0)`}
      />
      <circle
        cx={c}
        cy={c}
        r={r}
        fill="none"
        stroke={OUTLINE_COLOR}
        strokeWidth={0.75}
      />
    </svg>
  );
}
