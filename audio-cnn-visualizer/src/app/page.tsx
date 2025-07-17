"use client";

import { useState } from "react";
import Image from "next/image";
import ColorScale from "~/components/ColorScale";
import FeatureMap from "~/components/FeatureMap";
import LoadingAnimation from "~/components/LoadingAnimation";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Progress } from "~/components/ui/progress";
import Waveform from "~/components/Waveform";

interface Prediction {
  class: string;
  confidence: number;
}

interface LayerData {
  shape: number[];
  values: number[][];
}

type VisualizationData = Record<string, LayerData>;

interface WaveformData {
  values: number[];
  sample_rate: number;
  duration: number;
}

interface ApiResponse {
  predictions: Prediction[];
  visualization: VisualizationData;
  input_spectrogram: LayerData;
  waveform: WaveformData;
}

const ESC50_EMOJI_MAP: Record<string, string> = {
  dog: "🐕",
  rain: "🌧️",
  crying_baby: "👶",
  door_wood_knock: "🚪",
  helicopter: "🚁",
  rooster: "🐓",
  sea_waves: "🌊",
  sneezing: "🤧",
  mouse_click: "🖱️",
  chainsaw: "🪚",
  pig: "🐷",
  crackling_fire: "🔥",
  clapping: "👏",
  keyboard_typing: "⌨️",
  siren: "🚨",
  cow: "🐄",
  crickets: "🦗",
  breathing: "💨",
  door_wood_creaks: "🚪",
  car_horn: "📯",
  frog: "🐸",
  chirping_birds: "🐦",
  coughing: "😷",
  can_opening: "🥫",
  engine: "🚗",
  cat: "🐱",
  water_drops: "💧",
  footsteps: "👣",
  washing_machine: "🧺",
  train: "🚂",
  hen: "🐔",
  wind: "💨",
  laughing: "😂",
  vacuum_cleaner: "🧹",
  church_bells: "🔔",
  insects: "🦟",
  pouring_water: "🚰",
  brushing_teeth: "🪥",
  clock_alarm: "⏰",
  airplane: "✈️",
  sheep: "🐑",
  toilet_flush: "🚽",
  snoring: "😴",
  clock_tick: "⏱️",
  fireworks: "🎆",
  crow: "🐦‍⬛",
  thunderstorm: "⛈️",
  drinking_sipping: "🥤",
  glass_breaking: "🔨",
  hand_saw: "🪚",
};

const getEmojiForClass = (className: string): string => {
  return ESC50_EMOJI_MAP[className] ?? "🔈";
};

function splitLayers(visualization: VisualizationData) {
  const main: [string, LayerData][] = [];
  const internals: Record<string, [string, LayerData][]> = {};

  for (const [name, data] of Object.entries(visualization)) {
    if (!name.includes(".")) {
      main.push([name, data]);
    } else {
      const [parent] = name.split(".");
      if (parent === undefined) continue;

      internals[parent] ??= [];
      internals[parent].push([name, data]);
    }
  }

  return { main, internals };
}

export default function HomePage() {
  const [vizData, setVizData] = useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setIsLoading(true);
    setError(null);
    setVizData(null);

    const reader = new FileReader();
    reader.readAsArrayBuffer(file);
    reader.onload = async () => {
      try {
        const arrayBuffer = reader.result as ArrayBuffer;
        const base64String = btoa(
          new Uint8Array(arrayBuffer).reduce(
            (data, byte) => data + String.fromCharCode(byte),
            "",
          ),
        );

        const response = await fetch("https://amartya-007--audio-cnn-inference-audioclassifier-inference.modal.run/", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ audio_data: base64String }),
        });

        if (!response.ok) {
          throw new Error(`API error ${response.statusText}`);
        }

        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
        const data: ApiResponse = await response.json();
        setVizData(data);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "An unknown error occured",
        );
      } finally {
        setIsLoading(false);
      }
    };
    reader.onerror = () => {
      setError("Failed ot read the file.");
      setIsLoading(false);
    };
  };

  const { main, internals } = vizData
    ? splitLayers(vizData?.visualization)
    : { main: [], internals: {} };

  return (
    <>
      {isLoading && <LoadingAnimation />}
      <main className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900 p-8">
      <div className="mx-auto max-w-[100%]">
        <div className="mb-12 text-center relative">
          <div className="absolute inset-0 bg-gradient-to-r from-purple-600/30 via-violet-600/30 to-fuchsia-600/30 blur-3xl -z-10"></div>
          <div className="mb-6 flex justify-center">
            <div className="rounded-2xl bg-gradient-to-r from-purple-600 to-violet-600 p-4 shadow-2xl animate-float border border-purple-400/50">
              <Image 
                src="/audio.png" 
                alt="Audio Visualizer Icon" 
                width={48}
                height={48}
                className="object-contain brightness-110"
              />
            </div>
          </div>
          <h1 className="mb-4 bg-gradient-to-r from-purple-400 to-violet-300 bg-clip-text text-5xl font-bold tracking-tight text-transparent drop-shadow-lg">
            CNN Audio Visualizer
          </h1>
          <p className="text-lg mb-8 text-gray-300 max-w-2xl mx-auto leading-relaxed">
            Upload a WAV file to see the model&apos;s predictions and feature maps with advanced AI visualization
          </p>

          <div className="flex flex-col items-center">
            <div className="relative inline-block group">
              <input
                type="file"
                accept=".wav"
                id="file-upload"
                onChange={handleFileChange}
                disabled={isLoading}
                className="absolute inset-0 w-full h-full cursor-pointer opacity-0 z-10"
              />
              <Button
                disabled={isLoading}
                className="relative border-2 border-purple-400/50 bg-gray-800/80 hover:bg-purple-800/80 text-purple-200 hover:text-white font-semibold px-8 py-4 rounded-xl shadow-xl hover:shadow-2xl transition-all duration-300 group-hover:scale-105 backdrop-blur-sm"
                variant="outline"
                size="lg"
                type="button"
              >
                {isLoading ? (
                  <div className="flex items-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-purple-400 border-t-transparent"></div>
                    Analyzing...
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <span className="text-xl">📁</span>
                    Choose Audio File
                  </div>
                )}
              </Button>
            </div>

            {fileName && (
              <Badge
                variant="secondary"
                className="mt-4 bg-gradient-to-r from-purple-800/80 to-violet-800/80 text-purple-200 border border-purple-400/50 px-4 py-2 rounded-full backdrop-blur-sm"
              >
                <span className="mr-2">🎵</span>
                {fileName}
              </Badge>
            )}
          </div>
        </div>

        {error && (
          <div className="mb-8 transform transition-all duration-500 ease-out">
            <Card className="border-red-400/50 bg-gradient-to-r from-red-900/80 to-pink-900/80 shadow-2xl backdrop-blur-sm">
              <CardContent className="p-6">
                <div className="flex items-center gap-3">
                  <span className="text-2xl">⚠️</span>
                  <p className="text-red-200 font-medium">Error: {error}</p>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {vizData && (
          <div className="space-y-8 animate-in fade-in duration-700">
            <Card className="shadow-2xl border border-purple-500/30 bg-gradient-to-r from-gray-800/90 to-purple-900/90 overflow-hidden backdrop-blur-sm">
              <CardHeader className="bg-gradient-to-r from-purple-700 to-violet-700 text-white border-b border-purple-500/30 px-4 py-3">
                <CardTitle className="flex items-center gap-2 text-xl font-semibold">
                  <span className="text-2xl">🎯</span>
                  Top Predictions
                </CardTitle>
              </CardHeader>
              <CardContent className="p-6">
                <div className="space-y-4">
                  {vizData.predictions.slice(0, 3).map((pred, i) => (
                    <div key={pred.class} className="space-y-3 p-4 rounded-lg bg-gradient-to-r from-gray-800/50 to-purple-800/50 border border-purple-400/30 hover:shadow-lg transition-all duration-300 hover:scale-[1.02] group backdrop-blur-sm">
                      <div className="flex items-center justify-between">
                        <div className="text-lg font-semibold text-gray-200 flex items-center gap-3">
                          <span className="text-2xl group-hover:scale-110 transition-transform duration-300">{getEmojiForClass(pred.class)}</span>
                          <span className="capitalize group-hover:text-white transition-colors duration-300">{pred.class.replaceAll("_", " ")}</span>
                        </div>
                        <Badge 
                          variant={i === 0 ? "default" : "secondary"}
                          className={`
                            ${i === 0 
                              ? "bg-gradient-to-r from-emerald-600 to-green-600 text-white shadow-lg animate-pulse" 
                              : i === 1 
                                ? "bg-gradient-to-r from-purple-500 to-violet-500 text-white"
                                : "bg-gradient-to-r from-gray-600 to-slate-600 text-white"
                            } px-3 py-1 text-sm font-bold group-hover:scale-105 transition-transform duration-300
                          `}
                        >
                          {(pred.confidence * 100).toFixed(1)}%
                        </Badge>
                      </div>
                      <div className="relative">
                        <Progress 
                          value={pred.confidence * 100} 
                          className="h-3 bg-gray-200 rounded-full overflow-hidden"
                        />
                        <div 
                          className="absolute top-0 left-0 h-full rounded-full transition-all duration-1000 ease-out"
                          style={{
                            width: `${pred.confidence * 100}%`,
                            background: `linear-gradient(to right, 
                              ${i === 0 ? '#10b981, #059669' : i === 1 ? '#3b82f6, #2563eb' : '#6b7280, #4b5563'})`
                          }}
                        />
                        {i === 0 && (
                          <div className="absolute inset-0 rounded-full bg-gradient-to-r from-transparent via-white/30 to-transparent -skew-x-12 animate-pulse"></div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
              <Card className="shadow-xl border-0 bg-gradient-to-br from-gray-800/80 to-gray-900/80 backdrop-blur-md overflow-hidden group hover:shadow-2xl transition-all duration-500">
                <CardHeader className="bg-gradient-to-r from-gray-700 to-gray-800 text-white px-4 py-3">
                  <CardTitle className="flex items-center gap-2 text-xl font-semibold">
                    <span className="text-2xl">📊</span>
                    Input Spectrogram
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-6">
                  <div className="rounded-lg overflow-hidden border-2 border-gray-600/30 shadow-inner bg-gray-900/40">
                    <FeatureMap
                      data={vizData.input_spectrogram.values}
                      title={`${vizData.input_spectrogram.shape.join(" x ")}`}
                      spectrogram
                    />
                  </div>

                  <div className="mt-6 flex justify-end">
                    <div className="bg-gray-900/60 backdrop-blur-sm rounded-lg p-2 shadow-md border border-gray-600/30">
                      <ColorScale width={200} height={16} min={-1} max={1} />
                    </div>
                  </div>
                </CardContent>
              </Card>
              <Card className="shadow-xl border-0 bg-gradient-to-br from-gray-800/80 to-gray-900/80 backdrop-blur-md overflow-hidden group hover:shadow-2xl transition-all duration-500">
                <CardHeader className="bg-gradient-to-r from-gray-700 to-gray-800 text-white px-4 py-3">
                  <CardTitle className="flex items-center gap-2 text-xl font-semibold">
                    <span className="text-2xl">🎵</span>
                    Audio Waveform
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-6">
                  <div className="rounded-lg overflow-hidden border-2 border-gray-600/30 shadow-inner bg-gray-900/40">
                    <Waveform
                      data={vizData.waveform.values}
                      title={`${vizData.waveform.duration.toFixed(2)}s * ${vizData.waveform.sample_rate}Hz`}
                    />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Feature maps */}
            <Card className="shadow-xl border-0 bg-gradient-to-br from-gray-800/80 to-gray-900/80 backdrop-blur-md overflow-hidden">
              <CardHeader className="bg-gradient-to-r from-gray-700 to-gray-800 text-white px-4 py-3">
                <CardTitle className="flex items-center gap-2 text-xl font-semibold">
                  <span className="text-2xl">🧠</span>
                  Convolutional Layer Outputs
                </CardTitle>
              </CardHeader>
              <CardContent className="p-6">
                <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-8">
                  {main.map(([mainName, mainData]) => (
                    <div key={mainName} className="space-y-4 group">
                      <div className="transform transition-all duration-300 hover:scale-105">
                        <div className="bg-gradient-to-r from-gray-800/80 to-gray-700/80 backdrop-blur-sm rounded-lg p-4 border border-purple-400/30 shadow-md hover:shadow-lg transition-all duration-300">
                          <h4 className="mb-3 font-bold text-purple-300 text-center capitalize">
                            {mainName}
                          </h4>
                          <div className="rounded-lg overflow-hidden border border-purple-400/20 shadow-sm">
                            <FeatureMap
                              data={mainData.values}
                              title={`${mainData.shape.join(" x ")}`}
                            />
                          </div>
                        </div>
                      </div>

                      {internals[mainName] && (
                        <div className="h-80 overflow-y-auto rounded-lg border-2 border-purple-400/30 bg-gradient-to-b from-gray-800/60 to-gray-900/80 backdrop-blur-sm p-3 shadow-inner">
                          <div className="space-y-3">
                            {internals[mainName]
                              .sort(([a], [b]) => a.localeCompare(b))
                              .map(([layerName, layerData]) => (
                                <div key={layerName} className="bg-gray-800/60 backdrop-blur-sm rounded-md p-2 border border-purple-400/20 shadow-sm hover:shadow-md transition-all duration-200">
                                  <FeatureMap
                                    data={layerData.values}
                                    title={layerName.replace(`${mainName}.`, "")}
                                    internal={true}
                                  />
                                </div>
                              ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
                <div className="mt-8 flex justify-center">
                  <div className="bg-gray-900/60 backdrop-blur-sm rounded-lg p-3 shadow-lg border-2 border-purple-400/30">
                    <ColorScale width={200} height={16} min={-1} max={1} />
                    <p className="text-sm text-purple-300 text-center mt-2 font-medium">Feature Intensity Scale</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </main>
    </>
  );
}
