// components/ui/alerts-panel.tsx
import { Alert } from '@/app/dashboard/page'; // Adjust path if needed

interface Props {
  alerts: Alert[];
}

export function AlertsPanel({ alerts }: Props) {
  return (
    <div className="bg-gray-800 p-6 rounded-lg h-96 overflow-y-auto">
      <h2 className="text-xl font-bold mb-4">Real-time Alerts</h2>
      <ul className="space-y-3">
        {alerts.length > 0 ? alerts.map((alert, index) => (
          <li key={index} className={`p-3 rounded-md ${alert.level === 'HIGH' ? 'bg-red-500/30 border-l-4 border-red-500' : 'bg-orange-500/30 border-l-4 border-orange-500'}`}>
            <div className="flex justify-between items-center">
              <span className="font-bold">{alert.level}</span>
              <span className="text-xs text-gray-400">{new Date(alert.timestamp).toLocaleTimeString()}</span>
            </div>
            <p className="text-sm">High density in {alert.zone_name}</p>
          </li>
        )) : (
          <p className="text-gray-500">No alerts triggered yet.</p>
        )}
      </ul>
    </div>
  );
}