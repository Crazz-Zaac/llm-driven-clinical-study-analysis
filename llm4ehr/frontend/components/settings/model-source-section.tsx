"use client"

import { useAppStore } from "@/lib/store"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Cpu, Cloud, Info } from "lucide-react"

export function ModelSourceSection() {
  const { useHostedModels, setUseHostedModels } = useAppStore()

  return (
    <Card>
      <CardHeader>
        <CardTitle>Model Source</CardTitle>
        <CardDescription>
          Choose whether to use models running locally on your device or hosted models accessed via API.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Improved Toggle UI */}
        <div className="inline-flex rounded-lg border border-muted-foreground/20 bg-muted/30 p-1">
          <Button
            variant={!useHostedModels ? "default" : "ghost"}
            size="sm"
            onClick={() => setUseHostedModels(false)}
            className="gap-2 rounded-md px-4"
          >
            <Cpu className="h-4 w-4" />
            Local Models
          </Button>
          <Button
            variant={useHostedModels ? "default" : "ghost"}
            size="sm"
            onClick={() => setUseHostedModels(true)}
            className="gap-2 rounded-md px-4"
          >
            <Cloud className="h-4 w-4" />
            Hosted Models
          </Button>
        </div>

        {useHostedModels ? (
          <div className="space-y-3 p-4 rounded-lg bg-accent/5 border border-accent/20">
            <h4 className="font-semibold text-sm flex items-center gap-2">
              <Cloud className="h-4 w-4" />
              Hosted Models
            </h4>
            <ul className="text-sm text-muted-foreground space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-primary">+</span>
                <span>Access to powerful models</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary">+</span>
                <span>No local compute required</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-muted-foreground">-</span>
                <span>Requires API key and may have costs</span>
              </li>
            </ul>
          </div>
        ) : (
          <div className="space-y-3 p-4 rounded-lg bg-primary/5 border border-primary/20">
            <h4 className="font-semibold text-sm flex items-center gap-2">
              <Cpu className="h-4 w-4" />
              Local Models
            </h4>
            <ul className="text-sm text-muted-foreground space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-primary">+</span>
                <span>Full privacy - data stays on device</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary">+</span>
                <span>No API costs</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-muted-foreground">-</span>
                <span>Requires download and local compute</span>
              </li>
            </ul>
          </div>
        )}

        <div className="flex items-start gap-2 rounded-lg bg-muted/50 p-3 text-sm">
          <Info className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
          <p className="text-muted-foreground">
            {useHostedModels
              ? "Configure your API key below to access hosted models. Your key is stored securely in your browser."
              : "Download and configure local models below. Models run entirely on your device for maximum privacy."}
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
