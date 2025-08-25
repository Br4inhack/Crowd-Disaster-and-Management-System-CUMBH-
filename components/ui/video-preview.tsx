// components/ui/video-preview.tsx
interface Props {
    videoUrl: string | null;
    progress: number;
  }
  
  export function VideoPreview({ videoUrl, progress }: Props) {
    return (
      <div className="bg-gray-800 p-6 rounded-lg">
        <h2 className="text-xl font-bold mb-4">2. Analyzed Video Preview</h2>
        <div className="aspect-video bg-black rounded flex items-center justify-center">
          {videoUrl ? (
            <video src={videoUrl} controls autoPlay className="w-full h-full rounded"></video>
          ) : (
            <div className="w-full p-8 text-center">
              <p className="mb-4">Analysis in progress...</p>
              <div className="w-full bg-gray-700 rounded-full h-2.5">
                <div className="bg-indigo-600 h-2.5 rounded-full" style={{ width: `${progress}%` }}></div>
              </div>
              <p className="mt-2 text-sm">{Math.round(progress)}%</p>
            </div>
          )}
        </div>
      </div>
    );
  }