"use client"

import { useAppStore } from "@/lib/store"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { RotateCcw, Info, Thermometer, Layers, Hash } from "lucide-react"

export function ParametersSection() {
  const {
    temperature,
    maxTokens,
    topP,
    setTemperature,
    setMaxTokens,
    setTopP,
  } = useAppStore()

  const handleReset = () => {
    setTemperature(0.7)
    setMaxTokens(2048)
    setTopP(0.9)
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Model Parameters</CardTitle>
              <CardDescription>
                Adjust these parameters to control how the model generates responses. 
                These settings affect creativity, consistency, and response length.
              </CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={handleReset}>
              <RotateCcw className="h-4 w-4 mr-2" />
              Reset
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-8">
          {/* Temperature */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label className="flex items-center gap-2">
                <Thermometer className="h-4 w-4" />
                Temperature
              </Label>
              <span className="text-sm font-mono bg-muted px-2 py-1 rounded">
                {temperature.toFixed(2)}
              </span>
            </div>
            <Slider
              value={[temperature]}
              onValueChange={([value]) => setTemperature(value)}
              min={0}
              max={2}
              step={0.01}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>0.0 - Focused</span>
              <span>1.0 - Balanced</span>
              <span>2.0 - Creative</span>
            </div>
            <div className="flex items-start gap-2 rounded-lg bg-muted/50 p-3 text-sm">
              <Info className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
              <p className="text-muted-foreground">
                <strong>Temperature</strong> controls randomness in responses. Lower values (0-0.3) produce 
                more deterministic, focused outputs ideal for factual queries. Higher values (0.7-2.0) 
                increase creativity and variation, useful for brainstorming or creative writing.
              </p>
            </div>
          </div>

          {/* Top P */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label className="flex items-center gap-2">
                <Layers className="h-4 w-4" />
                Top P (Nucleus Sampling)
              </Label>
              <span className="text-sm font-mono bg-muted px-2 py-1 rounded">
                {topP.toFixed(2)}
              </span>
            </div>
            <Slider
              value={[topP]}
              onValueChange={([value]) => setTopP(value)}
              min={0}
              max={1}
              step={0.01}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>0.0 - Narrow</span>
              <span>0.5 - Moderate</span>
              <span>1.0 - Full range</span>
            </div>
            <div className="flex items-start gap-2 rounded-lg bg-muted/50 p-3 text-sm">
              <Info className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
              <p className="text-muted-foreground">
                <strong>Top P</strong> (nucleus sampling) limits token selection to a cumulative probability. 
                A value of 0.9 means only tokens comprising the top 90% probability mass are considered. 
                Lower values make outputs more focused; higher values allow more diversity.
              </p>
            </div>
          </div>

          {/* Max Tokens */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label htmlFor="max-tokens" className="flex items-center gap-2">
                <Hash className="h-4 w-4" />
                Max Tokens
              </Label>
              <Input
                id="max-tokens"
                type="number"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value) || 2048)}
                className="w-28 text-right font-mono"
                min={1}
                max={32768}
              />
            </div>
            <Slider
              value={[maxTokens]}
              onValueChange={([value]) => setMaxTokens(value)}
              min={256}
              max={8192}
              step={256}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>256 - Brief</span>
              <span>2048 - Standard</span>
              <span>8192 - Extended</span>
            </div>
            <div className="flex items-start gap-2 rounded-lg bg-muted/50 p-3 text-sm">
              <Info className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
              <p className="text-muted-foreground">
                <strong>Max Tokens</strong> sets the maximum length of the generated response. 
                One token is roughly 4 characters or 0.75 words. Higher values allow longer responses 
                but may increase processing time and costs for hosted models.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Configuration Preview */}
      <Card>
        <CardHeader>
          <CardTitle>Current Configuration</CardTitle>
          <CardDescription>
            This JSON represents your current parameter settings. These values are automatically 
            saved and will be used for all conversations.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="rounded-lg bg-muted p-4 font-mono text-sm overflow-x-auto">
            <pre className="text-foreground">
{`{
  "temperature": ${temperature},
  "top_p": ${topP},
  "max_tokens": ${maxTokens}
}`}
            </pre>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
