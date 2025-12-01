import { Badge } from '@/components/ui/badge'

export const DeviceStatusPanel = ({ devices }) => (
  <div className="space-y-3">
    {devices.length === 0 ? (
      <p className="text-muted-foreground text-center py-8">No devices connected</p>
    ) : (
      devices.map((device) => (
        <div key={device.id} className="flex items-center justify-between p-3 bg-muted rounded-lg">
          <div>
            <p className="font-medium">{device.id}</p>
            <p className="text-sm text-muted-foreground">{device.platform} • {device.status}</p>
          </div>
          <Badge 
            variant={device.status === 'active' ? 'default' : 'secondary'}
            className="capitalize"
          >
            {device.status}
          </Badge>
        </div>
      ))
    )}
  </div>
)
