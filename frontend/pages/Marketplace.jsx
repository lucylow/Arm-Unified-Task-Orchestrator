import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Store, Star, Download, Search } from 'lucide-react'

export default function Marketplace() {
  const [searchTerm, setSearchTerm] = useState('')
  
  const plugins = [
    {
      id: 1,
      name: 'Email Automation',
      description: 'Automate email sending and management',
      author: 'AutoRL Team',
      rating: 4.8,
      downloads: 1243,
      price: 'Free',
      verified: true
    },
    {
      id: 2,
      name: 'Social Media Manager',
      description: 'Schedule and manage social media posts',
      author: 'Community',
      rating: 4.6,
      downloads: 892,
      price: '$9.99',
      verified: true
    },
    {
      id: 3,
      name: 'Screenshot Tool',
      description: 'Advanced screenshot and image processing',
      author: 'DevTools Inc',
      rating: 4.9,
      downloads: 2156,
      price: 'Free',
      verified: false
    },
    {
      id: 4,
      name: 'Data Export',
      description: 'Export automation results to various formats',
      author: 'AutoRL Team',
      rating: 4.7,
      downloads: 1567,
      price: '$4.99',
      verified: true
    }
  ]

  const filteredPlugins = plugins.filter(plugin =>
    plugin.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    plugin.description.toLowerCase().includes(searchTerm.toLowerCase())
  )

  return (
    <div className="container mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Plugin Marketplace</h1>
        <p className="text-muted-foreground">
          Extend AutoRL with community plugins and integrations
        </p>
      </div>

      <div className="mb-6">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search plugins..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredPlugins.map((plugin) => (
          <Card key={plugin.id}>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Store className="w-5 h-5" />
                  <span>{plugin.name}</span>
                </div>
                {plugin.verified && (
                  <Badge variant="default" className="ml-2">Verified</Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-4">
                {plugin.description}
              </p>
              <div className="space-y-2 mb-4">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Author</span>
                  <span className="font-medium">{plugin.author}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Rating</span>
                  <div className="flex items-center space-x-1">
                    <Star className="w-4 h-4 fill-yellow-500 text-yellow-500" />
                    <span className="font-medium">{plugin.rating}</span>
                  </div>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Downloads</span>
                  <div className="flex items-center space-x-1">
                    <Download className="w-4 h-4" />
                    <span className="font-medium">{plugin.downloads}</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-lg font-bold text-primary">{plugin.price}</span>
                <Button size="sm">Install</Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
