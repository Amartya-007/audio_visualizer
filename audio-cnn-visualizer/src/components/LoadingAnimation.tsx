const LoadingAnimation = () => {
  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-[9999] backdrop-blur-lg">
      <div className="relative bg-gradient-to-br from-gray-900/95 via-purple-900/95 to-violet-900/95 backdrop-blur-xl rounded-3xl p-10 shadow-2xl max-w-md w-full mx-4 border border-purple-400/40 overflow-hidden">
        {/* Animated background elements */}
        <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 via-violet-600/20 to-fuchsia-600/20 blur-3xl animate-pulse"></div>
        <div className="absolute -top-10 -left-10 w-20 h-20 bg-purple-500/30 rounded-full blur-xl animate-bounce"></div>
        <div className="absolute -bottom-10 -right-10 w-16 h-16 bg-violet-500/30 rounded-full blur-xl animate-bounce" style={{animationDelay: '1s'}}></div>
        
        <div className="relative text-center">
          <div className="mb-8">
            <div className="relative mx-auto w-24 h-24">
              {/* Outer spinning ring */}
              <div className="absolute inset-0 rounded-full border-4 border-purple-300/30"></div>
              <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-purple-400 border-r-purple-400 animate-spin"></div>
              
              {/* Middle spinning ring */}
              <div className="absolute inset-2 rounded-full border-3 border-violet-300/30"></div>
              <div className="absolute inset-2 rounded-full border-3 border-transparent border-t-violet-500 border-r-violet-500 animate-spin" style={{animationDirection: 'reverse', animationDuration: '1.2s'}}></div>
              
              {/* Inner spinning ring */}
              <div className="absolute inset-4 rounded-full border-2 border-fuchsia-300/30"></div>
              <div className="absolute inset-4 rounded-full border-2 border-transparent border-t-fuchsia-400 border-r-fuchsia-400 animate-spin" style={{animationDuration: '0.8s'}}></div>
              
              {/* Center icon with glow */}
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                <div className="text-3xl animate-pulse drop-shadow-lg">🎵</div>
                <div className="absolute inset-0 text-3xl animate-ping opacity-30">🎵</div>
              </div>
            </div>
          </div>
          
          <h3 className="text-2xl font-bold bg-gradient-to-r from-purple-300 to-violet-200 bg-clip-text text-transparent mb-3">
            Analyzing Audio
          </h3>
          
          <p className="text-purple-200/80 mb-8 text-sm leading-relaxed">
            Our AI is processing your audio file and generating beautiful visualizations...
          </p>
          
          {/* Enhanced progress steps */}
          <div className="space-y-4 mb-8">
            <div className="flex justify-between items-center p-3 rounded-lg bg-gray-800/50 border border-purple-400/20">
              <span className="text-sm text-purple-200 flex items-center gap-2">
                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                Extracting features
              </span>
              <span className="text-green-400">✓</span>
            </div>
            <div className="flex justify-between items-center p-3 rounded-lg bg-gray-800/50 border border-purple-400/20 bg-gradient-to-r from-purple-800/30 to-violet-800/30">
              <span className="text-sm text-white flex items-center gap-2">
                <span className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></span>
                Running CNN model
              </span>
              <div className="w-5 h-5 border-2 border-purple-400 border-t-transparent rounded-full animate-spin"></div>
            </div>
            <div className="flex justify-between items-center p-3 rounded-lg bg-gray-800/30 border border-purple-400/10">
              <span className="text-sm text-purple-400/60 flex items-center gap-2">
                <span className="w-2 h-2 bg-gray-500 rounded-full"></span>
                Generating visualizations
              </span>
              <span className="text-purple-400/60">⏳</span>
            </div>
          </div>
          
          {/* Enhanced progress bar */}
          <div className="relative w-full bg-gray-800/50 rounded-full h-3 overflow-hidden border border-purple-400/20">
            <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 to-violet-600/20"></div>
            <div 
              className="h-full bg-gradient-to-r from-purple-500 via-violet-500 to-fuchsia-500 rounded-full relative overflow-hidden"
              style={{width: '65%'}}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-ping"></div>
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse"></div>
            </div>
          </div>
          
          <div className="mt-3 text-xs text-purple-300/70">Processing... 65%</div>
        </div>
      </div>
    </div>
  );
};

export default LoadingAnimation;
