// components/ui/insights-panel.tsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Insight } from '@/app/dashboard/page'; // Adjust path if needed

interface Props {
  insights: Insight[];
}

export function InsightsPanel({ insights }: Props) {
  return (
    <div className="bg-gray-800 p-6 rounded-lg">
      <h2 className="text-xl font-bold mb-4">Live Insights</h2>
      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer>
          <LineChart data={insights}>
            <CartesianGrid strokeDasharray="3 3" stroke="#4A5568" />
            <XAxis dataKey="frame" stroke="#A0AEC0" />
            <YAxis stroke="#A0AEC0" />
            <Tooltip contentStyle={{ backgroundColor: '#1A202C', border: 'none' }} />
            <Legend />
            <Line type="monotone" dataKey="crowd_count" stroke="#818CF8" strokeWidth={2} name="Crowd Count" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}