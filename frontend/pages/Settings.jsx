import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Settings as SettingsIcon, Shield, Bell, Zap } from 'lucide-react'

export default function Settings() {
  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Settings</h1>
        <p className="text-muted-foreground">
          Configure your AutoRL platform preferences
        </p>
      </div>

      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <SettingsIcon className="w-5 h-5 mr-2 text-primary" />
              General Settings
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium mb-2 block">API Endpoint</label>
              <Input defaultValue="http://localhost:5000/api" />
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Default Device</label>
              <Input placeholder="Select default device" />
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Max Task Steps</label>
              <Input type="number" defaultValue="50" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Zap className="w-5 h-5 mr-2 text-primary" />
              AI & Training
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Enable RL Training</p>
                <p className="text-sm text-muted-foreground">Automatically improve task execution</p>
              </div>
              <Switch defaultChecked />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Auto-save Models</p>
                <p className="text-sm text-muted-foreground">Save trained models automatically</p>
              </div>
              <Switch defaultChecked />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Debug Mode</p>
                <p className="text-sm text-muted-foreground">Show detailed execution logs</p>
              </div>
              <Switch />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Shield className="w-5 h-5 mr-2 text-primary" />
              Security & Privacy
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">PII Data Masking</p>
                <p className="text-sm text-muted-foreground">Automatically mask sensitive information</p>
              </div>
              <Switch defaultChecked />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Safety Guardrails</p>
                <p className="text-sm text-muted-foreground">Prevent harmful actions</p>
              </div>
              <Switch defaultChecked />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Bell className="w-5 h-5 mr-2 text-primary" />
              Notifications
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Task Completion</p>
                <p className="text-sm text-muted-foreground">Notify when tasks complete</p>
              </div>
              <Switch defaultChecked />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Device Disconnected</p>
                <p className="text-sm text-muted-foreground">Alert on device disconnection</p>
              </div>
              <Switch defaultChecked />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Training Complete</p>
                <p className="text-sm text-muted-foreground">Notify when training finishes</p>
              </div>
              <Switch />
            </div>
          </CardContent>
        </Card>

        <div className="flex justify-end space-x-4">
          <Button variant="outline">Reset to Defaults</Button>
          <Button>Save Changes</Button>
        </div>
      </div>
    </div>
  )
}
