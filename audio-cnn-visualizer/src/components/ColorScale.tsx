const ColorScale = ({
  width = 200,
  height = 16,
  min = -1,
  max = 1,
}: {
  width?: number;
  height?: number;
  min?: number;
  max?: number;
}) => {
  return (
    <div className="flex items-center gap-4">
      <span className="text-sm font-semibold text-slate-600 bg-slate-100 px-2 py-1 rounded">{min}</span>
      <div
        className="rounded-lg border-2 border-slate-300 shadow-md"
        style={{
          width: `${width}px`,
          height: `${height}px`,
          background:
            "linear-gradient(to right, rgb(255, 128, 51), rgb(255, 255, 255), rgb(51,128, 255))",
        }}
      />
      <span className="text-sm font-semibold text-slate-600 bg-slate-100 px-2 py-1 rounded">{max}</span>
    </div>
  );
};

export default ColorScale;
