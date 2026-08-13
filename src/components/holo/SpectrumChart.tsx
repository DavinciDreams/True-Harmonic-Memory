// Small bar chart for a Gegenbauer harmonic spectrum: no chart library
// needed, just SVG rects scaled to the coefficient magnitudes.
export default function SpectrumChart({
  spectrum,
  height = 64,
}: {
  spectrum: number[];
  height?: number;
}) {
  const max = Math.max(...spectrum.map((v) => Math.abs(v)), 1e-6);
  const barW = 100 / spectrum.length;
  return (
    <svg viewBox={`0 0 100 ${height}`} className="w-full" style={{ height }}>
      <line x1={0} y1={height / 2} x2={100} y2={height / 2} stroke="#334155" strokeWidth={0.5} />
      {spectrum.map((v, i) => {
        const h = (Math.abs(v) / max) * (height / 2 - 2);
        const x = i * barW + barW * 0.15;
        const w = barW * 0.7;
        const y = v >= 0 ? height / 2 - h : height / 2;
        return (
          <rect
            key={i}
            x={x}
            y={y}
            width={w}
            height={Math.max(h, 0.5)}
            fill={v >= 0 ? "#5eb4ff" : "#ff8a65"}
            rx={0.6}
          />
        );
      })}
    </svg>
  );
}
