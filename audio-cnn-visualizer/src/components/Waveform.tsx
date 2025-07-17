const Waveform = ({ data, title }: { data: number[]; title: string }) => {
  if (!data || data.length === 0) return null;

  const width = 600;
  const height = 300;
  const centerY = height / 2;

  const validData = data.filter((val) => !isNaN(val) && isFinite(val));
  if (validData.length === 0) return null;

  const min = Math.min(...validData);
  const max = Math.max(...validData);
  const range = max - min;
  const scaleY = height * 0.45;

  const pathData = validData
    .map((sample, i) => {
      const x = (i / (validData.length - 1)) * width;
      let y = centerY;

      if (range > 0) {
        const normalizedSample = (sample - min) / range; // 0 - 1, -0.5 - 0.5
        y = centerY - (normalizedSample - 0.5) * 2 * scaleY;
      }

      return `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");

  return (
    <div className="flex h-full w-full flex-col">
      <div className="flex flex-1 items-center justify-center">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="xMidYMid meet"
          className="block max-h-[300px] max-w-full rounded-lg border-2 border-green-200 shadow-sm transition-all duration-300 hover:shadow-md"
        >
          {/* Grid lines for better visualization */}
          <defs>
            <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
              <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#f0f9ff" strokeWidth="1"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
          
          {/* Center line */}
          <path
            d={`M 0 ${centerY} H ${width}`}
            stroke="#86efac"
            strokeWidth="2"
            strokeDasharray="5,5"
          />
          
          {/* Waveform with gradient */}
          <defs>
            <linearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#10b981" />
              <stop offset="25%" stopColor="#06d6a0" />
              <stop offset="50%" stopColor="#14b8a6" />
              <stop offset="75%" stopColor="#0891b2" />
              <stop offset="100%" stopColor="#0284c7" />
            </linearGradient>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
              <feMerge> 
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>
          
          <path
            d={pathData}
            fill="none"
            stroke="url(#waveGradient)"
            strokeWidth="2.5"
            strokeLinejoin="round"
            strokeLinecap="round"
            filter="url(#glow)"
            className="drop-shadow-lg"
          />
        </svg>
      </div>
      <p className="mt-3 text-center text-sm font-medium text-green-700 bg-green-50 rounded-full px-3 py-1 border border-green-200">
        {title}
      </p>
    </div>
  );
};

export default Waveform;
