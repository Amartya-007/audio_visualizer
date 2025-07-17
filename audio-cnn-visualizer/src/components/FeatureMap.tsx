import { getColor } from "~/lib/colors";
import { useState } from "react";

const FeatureMap = ({
  data,
  title,
  internal,
  spectrogram,
}: {
  data: number[][];
  title: string;
  internal?: boolean;
  spectrogram?: boolean;
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const [hoveredPixel, setHoveredPixel] = useState<{x: number, y: number, value: number} | null>(null);
  
  if (!data?.length || !data[0]?.length) return null;

  const mapHeight = data.length;
  const mapWidth = data[0].length;

  const absMax = data
    .flat()
    .reduce((acc, val) => Math.max(acc, Math.abs(val ?? 0)), 0);

  const handlePixelHover = (i: number, j: number, value: number) => {
    setHoveredPixel({ x: j, y: i, value });
  };

  return (
    <div className="w-full text-center relative group">
      <div 
        className="relative overflow-hidden"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => {
          setIsHovered(false);
          setHoveredPixel(null);
        }}
      >
        <svg
          viewBox={`0 0 ${mapWidth} ${mapHeight}`}
          preserveAspectRatio="none"
          className={`mx-auto block rounded-lg border-2 shadow-lg transition-all duration-500 ${
            isHovered ? 'shadow-2xl scale-105' : 'shadow-lg'
          } ${
            internal 
              ? "w-full max-w-32 border-gray-300 hover:border-gray-400" 
              : spectrogram 
                ? "w-full object-contain border-purple-300 hover:border-purple-400" 
                : "max-h-[300px] w-full max-w-[500px] object-contain border-indigo-300 hover:border-indigo-400"
          }`}
        >
          {/* Background glow effect */}
          <defs>
            <filter id={`glow-${title.replace(/\s+/g, '-')}`}>
              <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
              <feMerge> 
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
            
            {/* Gradient overlay for better depth */}
            <linearGradient id={`overlay-${title.replace(/\s+/g, '-')}`} x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="rgba(255,255,255,0.1)" />
              <stop offset="100%" stopColor="rgba(0,0,0,0.1)" />
            </linearGradient>
          </defs>
          
          {data.flatMap((row, i) =>
            row.map((value, j) => {
              const normalizedValues = absMax === 0 ? 0 : value / absMax;
              const [r, g, b] = getColor(normalizedValues);
              const isHoveredPixel = hoveredPixel?.x === j && hoveredPixel?.y === i;
              
              return (
                <rect
                  key={`${i}-${j}`}
                  x={j}
                  y={i}
                  width={1}
                  height={1}
                  fill={`rgb(${r},${g},${b})`}
                  filter={isHoveredPixel ? `url(#glow-${title.replace(/\s+/g, '-')})` : undefined}
                  opacity={isHovered && !isHoveredPixel ? 0.7 : 1}
                  onMouseEnter={() => !internal && handlePixelHover(i, j, value)}
                  className="transition-opacity duration-200 cursor-crosshair"
                />
              );
            }),
          )}
          
          {/* Overlay gradient for depth effect */}
          <rect
            width="100%"
            height="100%"
            fill={`url(#overlay-${title.replace(/\s+/g, '-')})`}
            pointerEvents="none"
          />
        </svg>
        
        {/* Hover tooltip for non-internal feature maps */}
        {hoveredPixel && !internal && (
          <div className="absolute top-2 left-2 bg-black bg-opacity-80 text-white text-xs px-2 py-1 rounded shadow-lg pointer-events-none z-10">
            <div>Position: ({hoveredPixel.x}, {hoveredPixel.y})</div>
            <div>Value: {hoveredPixel.value.toFixed(4)}</div>
          </div>
        )}
        
        {/* Scanning line animation for spectrogram */}
        {spectrogram && isHovered && (
          <div className="absolute inset-0 pointer-events-none">
            <div className="w-0.5 h-full bg-white opacity-60 animate-pulse absolute left-1/2 transform -translate-x-1/2 shadow-lg"></div>
          </div>
        )}
      </div>
      
      <div className={`mt-3 px-2 py-1 rounded-lg transition-all duration-300 ${
        isHovered ? 'transform scale-105' : ''
      }`}>
        <p className={`text-sm font-bold ${
          internal 
            ? "text-gray-600 bg-gray-100" 
            : spectrogram 
              ? "text-purple-700 bg-purple-100" 
              : "text-indigo-700 bg-indigo-100"
        } px-3 py-1 rounded-full inline-block shadow-sm`}>
          {title}
        </p>
        
        {!internal && (
          <div className={`text-xs mt-1 ${
            spectrogram ? "text-purple-600" : "text-indigo-600"
          }`}>
            <span className="opacity-75">
              {mapWidth} × {mapHeight} | Max: {absMax.toFixed(3)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

export default FeatureMap;
