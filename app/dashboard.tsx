// app/dashboard/page.tsx
'use client';

import { useState, useEffect, useRef } from 'react';
import { useSearchParams } from 'next/navigation';
import { VideoUpload } from '@/components/ui/video-upload';
import { VideoPreview } from '@/components/ui/video-preview';
import { InsightsPanel } from '@/components/ui/insights-panel';
import { AlertsPanel } from '@/components/ui/alerts-panel';

export interface Alert {
  level: 'MEDIUM' | 'HIGH';
  timestamp: string;
  zone_name: string;
}

export interface Insight {
  frame: number;
  crowd_count: number;
}

export default function Dashboard() {
  const searchParams = useSearchParams();
  const role = searchParams.get('role');
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [insights, setInsights] = useState<Insight[]>([]);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    ws.current = new WebSocket('ws://localhost:8000/ws');

    ws.current.onopen = () => console.log('WebSocket connected');
    ws.current.onclose = () => console.log('WebSocket disconnected');
    
    ws.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'progress') {
        setProgress(message.progress);
        setInsights(prev => [...prev, { frame: message.frame, crowd_count: message.crowd_count }]);
      } else if (message.type === 'alerts') {
        setAlerts(prev => [...message.data, ...prev]);
      } else if (message.type === 'complete') {
        setVideoUrl(`http://localhost:8000${message.video_url}`);
        setProgress(100);
      }
    };

    return () => {
      ws.current?.close();
    };
  }, []);

  const handleUploadSuccess = () => {
    // Reset state for new analysis
    setVideoUrl(null);
    setProgress(0);
    setAlerts([]);
    setInsights([]);
    // Signal backend to start analysis
    ws.current?.send('start_analysis');
  };

  return (
    <main className="min-h-screen bg-gray-900 text-white p-8">
      <header className="mb-8">
        <h1 className="text-3xl font-bold">Crowd Management Dashboard</h1>
        <p className="text-indigo-400">Authority: {role}</p>
      </header>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-8">
          <VideoUpload onUploadSuccess={handleUploadSuccess} />
          <VideoPreview videoUrl={videoUrl} progress={progress} />
        </div>
        <div className="space-y-8">
          <AlertsPanel alerts={alerts} />
          <InsightsPanel insights={insights} />
        </div>
      </div>
    </main>
  );
}