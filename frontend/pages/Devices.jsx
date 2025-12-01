import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Smartphone, Battery, Wifi, Plus } from 'lucide-react'

export default function Devices() {
  const [devices] = useState([
    {
      id: 'android_pixel_7',
      name: 'Pixel 7',
      platform: 'Android',
      version: '14',
      status: 'connected',
      battery: 85,
      connection: 'USB'
    },
    {
      id: 'iphone_14',
      name: 'iPhone 14',
      platform: 'iOS',
      version: '17.2',
      status: 'connected',
      battery: 92,
      connection: 'Wi-Fi'
    },
    {
      id: 'samsung_s23',
      name: 'Galaxy S23',
      platform: 'Android',
      version: '14',
      status: 'disconnected',
      battery: 45,
      connection: 'None'
    }
  ])

  return (
    <div className="container mx-auto p-6">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold mb-2">Device Manager</h1>
          <p className="text-muted-foreground">
            Manage your connected Android and iOS devices
          </p>
        </div>
        <Button>
          <Plus className="w-4 h-4 mr-2" />
          Add Device
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {devices.map((device) => (
          <Card key={device.id}>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Smartphone className="w-5 h-5" />
                  <span>{device.name}</span>
                </div>
                <Badge variant={device.status === 'connected' ? 'default' : 'secondary'}>
                  {device.status}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Platform</span>
                  <span className="font-medium">{device.platform} {device.version}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Battery</span>
                  <div className="flex items-center space-x-1">
                    <Battery className="w-4 h-4" />
                    <span className="font-medium">{device.battery}%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Connection</span>
                  <div className="flex items-center space-x-1">
                    <Wifi className="w-4 h-4" />
                    <span className="font-medium">{device.connection}</span>
                  </div>
                </div>
                <div className="pt-2">
                  <Button variant="outline" className="w-full" size="sm">
                    View Details
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
