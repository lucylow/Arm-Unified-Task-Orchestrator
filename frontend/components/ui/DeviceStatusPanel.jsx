import { Badge } from '@/components/ui/badge'
import { Smartphone, Activity } from 'lucide-react'
import { cn } from '@/lib/utils'

export const DeviceStatusPanel = ({ devices }) => (
  <div className="space-y-3">
    {devices.length === 0 ? (
      <div className="text-center py-12 px-4">
        <div className="inline-flex p-3 rounded-full bg-muted/50 mb-3">
          <Smartphone className="w-6 h-6 text-muted-foreground" />
        </div>
        <p className="text-sm text-muted-foreground">No devices connected</p>
        <p className="text-xs text-muted-foreground/70 mt-1">
          Connect a device to start automation
        </p>
      </div>
    ) : (
      devices.map((device, index) => (
        <div 
          key={device.id} 
          className={cn(
            "group flex items-center justify-between p-4",
            "bg-gradient-to-br from-muted/50 to-muted/30",
            "rounded-lg border border-border/50",
            "transition-all duration-300",
            "hover:border-primary/30 hover:shadow-md hover:shadow-primary/5",
            "animate-fade-in"
          )}
          style={{ animationDelay: `${index * 0.1}s` }}
        >
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <div className={cn(
              "flex-shrink-0 p-2 rounded-lg",
              "bg-primary/10 text-primary",
              "group-hover:bg-primary/20 transition-colors"
            )}>
              <Smartphone className="w-4 h-4" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-medium text-foreground truncate">{device.id}</p>
              <div className="flex items-center gap-2 mt-0.5">
                <p className="text-xs text-muted-foreground">{device.platform}</p>
                <span className="text-muted-foreground/50">•</span>
                <div className="flex items-center gap-1">
                  <Activity className={cn(
                    "w-3 h-3",
                    device.status === 'active' ? "text-green-500 animate-pulse" : "text-muted-foreground"
                  )} />
                  <p className="text-xs text-muted-foreground capitalize">{device.status}</p>
                </div>
              </div>
            </div>
          </div>
          <Badge 
            variant={device.status === 'active' ? 'default' : 'secondary'}
            className={cn(
              "capitalize shrink-0 ml-2",
              device.status === 'active' && "bg-green-500 hover:bg-green-600"
            )}
          >
            {device.status}
          </Badge>
        </div>
      ))
    )}
  </div>
)
