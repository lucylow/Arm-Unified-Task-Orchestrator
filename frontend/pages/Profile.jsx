import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { User, Mail, Shield, Award } from 'lucide-react'

export default function Profile() {
  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">User Profile</h1>
        <p className="text-muted-foreground">
          Manage your account and preferences
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-1">
          <CardContent className="p-6">
            <div className="text-center">
              <div className="w-24 h-24 bg-primary/20 rounded-full flex items-center justify-center mx-auto mb-4">
                <User className="w-12 h-12 text-primary" />
              </div>
              <h2 className="text-xl font-bold mb-1">AutoRL User</h2>
              <p className="text-sm text-muted-foreground mb-4">user@autorl.com</p>
              <Badge>Premium Account</Badge>
            </div>
          </CardContent>
        </Card>

        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <User className="w-5 h-5 mr-2 text-primary" />
                Personal Information
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-2 block">Display Name</label>
                <Input defaultValue="AutoRL User" />
              </div>
              <div>
                <label className="text-sm font-medium mb-2 block">Email</label>
                <Input type="email" defaultValue="user@autorl.com" />
              </div>
              <div>
                <label className="text-sm font-medium mb-2 block">Organization</label>
                <Input placeholder="Your company name" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Award className="w-5 h-5 mr-2 text-primary" />
                Usage Statistics
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 border rounded-lg">
                  <p className="text-sm text-muted-foreground">Tasks Executed</p>
                  <p className="text-2xl font-bold">1,247</p>
                </div>
                <div className="p-3 border rounded-lg">
                  <p className="text-sm text-muted-foreground">Success Rate</p>
                  <p className="text-2xl font-bold">94.2%</p>
                </div>
                <div className="p-3 border rounded-lg">
                  <p className="text-sm text-muted-foreground">Models Trained</p>
                  <p className="text-2xl font-bold">12</p>
                </div>
                <div className="p-3 border rounded-lg">
                  <p className="text-sm text-muted-foreground">API Calls</p>
                  <p className="text-2xl font-bold">8,432</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Shield className="w-5 h-5 mr-2 text-primary" />
                Security
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-2 block">Current Password</label>
                <Input type="password" placeholder="••••••••" />
              </div>
              <div>
                <label className="text-sm font-medium mb-2 block">New Password</label>
                <Input type="password" placeholder="••••••••" />
              </div>
              <div>
                <label className="text-sm font-medium mb-2 block">Confirm Password</label>
                <Input type="password" placeholder="••••••••" />
              </div>
              <Button variant="outline" className="w-full">
                Change Password
              </Button>
            </CardContent>
          </Card>

          <div className="flex justify-end space-x-4">
            <Button variant="outline">Cancel</Button>
            <Button>Save Changes</Button>
          </div>
        </div>
      </div>
    </div>
  )
}
