import { useEffect, useState } from "react";

interface SuccessAnimationProps {
  show: boolean;
  onComplete: () => void;
}

const SuccessAnimation = ({ show, onComplete }: SuccessAnimationProps) => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (show) {
      setIsVisible(true);
      const timer = setTimeout(() => {
        setIsVisible(false);
        onComplete();
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [show, onComplete]);

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 flex items-center justify-center z-50 pointer-events-none">
      <div className="bg-white rounded-2xl p-8 shadow-2xl animate-fade-in-scale">
        <div className="text-center">
          <div className="mb-4">
            <div className="w-20 h-20 mx-auto bg-gradient-to-r from-green-400 to-emerald-500 rounded-full flex items-center justify-center animate-bounce">
              <span className="text-3xl text-white">✓</span>
            </div>
          </div>
          <h3 className="text-xl font-semibold text-gray-800 mb-2">
            Analysis Complete!
          </h3>
          <p className="text-gray-600">
            Your audio visualization is ready
          </p>
        </div>
      </div>
    </div>
  );
};

export default SuccessAnimation;
