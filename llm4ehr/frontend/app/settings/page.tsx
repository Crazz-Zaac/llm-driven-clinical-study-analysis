"use client"

import { Suspense } from "react"
import { useEffect, useState } from "react"
import Link from "next/link"
import { useSearchParams, useRouter } from "next/navigation"
import { useTheme } from "next-themes"
import { ArrowLeft, Cpu, Moon, Sun, Monitor, Palette } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { ModelSourceSection } from "@/components/settings/model-source-section"
import { LocalModelsSection } from "@/components/settings/local-models-section"
import { HostedModelsSection } from "@/components/settings/hosted-models-section"
import { EmbeddingModelSection } from "@/components/settings/embedding-model-section"
import { ParametersSection } from "@/components/settings/parameters-section"
import { ContextSection } from "@/components/settings/context-section"

function SettingsPageContent() {
  const { theme, setTheme } = useTheme()
  const router = useRouter()
  const searchParams = useSearchParams()
  const [activeTab, setActiveTab] = useState("models")
  const [isHydrated, setIsHydrated] = useState(false)

  // Set initial tab from URL on mount
  useEffect(() => {
    setIsHydrated(true)
    const tab = searchParams.get("tab") || "models"
    setActiveTab(tab)
  }, [searchParams])

  const handleTabChange = (newTab: string) => {
    setActiveTab(newTab)
    router.push(`/settings?tab=${newTab}`, { scroll: false })
  }

  if (!isHydrated) {
    return null
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="flex h-14 items-center justify-center gap-4 px-4 md:px-6">
          <Link href="/">
            <Button variant="ghost" size="icon" className="h-9 w-9 absolute left-4 md:left-6">
              <ArrowLeft className="h-4 w-4" />
              <span className="sr-only">Back to chat</span>
            </Button>
          </Link>
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary">
              <Cpu className="h-5 w-5 text-primary-foreground" />
            </div>
            <div className="flex flex-col">
              <span className="text-lg font-semibold tracking-tight">Settings</span>
              <span className="hidden text-xs text-muted-foreground sm:block">LLM4EHR Configuration</span>
            </div>
          </div>
        </div>
      </header>

      <main className="flex justify-center w-full py-8 px-4 md:px-6">
        <div className="w-full max-w-4xl">
          <Tabs value={activeTab} onValueChange={handleTabChange} className="space-y-6">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="models">Models</TabsTrigger>
              <TabsTrigger value="parameters">Parameters</TabsTrigger>
              <TabsTrigger value="context">Context</TabsTrigger>
              <TabsTrigger value="appearance">Appearance</TabsTrigger>
            </TabsList>

            <TabsContent value="models" className="space-y-6">
              <EmbeddingModelSection />
              <ModelSourceSection />
              <LocalModelsSection />
              <HostedModelsSection />
            </TabsContent>

            <TabsContent value="parameters" className="space-y-6">
              <ParametersSection />
            </TabsContent>

            <TabsContent value="context" className="space-y-6">
              <ContextSection />
            </TabsContent>

            <TabsContent value="appearance" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Palette className="h-5 w-5" />
                    Appearance
                  </CardTitle>
                  <CardDescription>
                    Choose your preferred theme. Your selection is automatically saved and will persist across sessions.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <RadioGroup
                    value={theme}
                    onValueChange={setTheme}
                    className="grid grid-cols-3 gap-4"
                  >
                    <div>
                      <RadioGroupItem value="light" id="light" className="peer sr-only" />
                      <Label
                        htmlFor="light"
                        className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary cursor-pointer"
                      >
                        <Sun className="mb-3 h-6 w-6" />
                        <span className="text-sm font-medium">Light</span>
                      </Label>
                    </div>
                    <div>
                      <RadioGroupItem value="dark" id="dark" className="peer sr-only" />
                      <Label
                        htmlFor="dark"
                        className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary cursor-pointer"
                      >
                        <Moon className="mb-3 h-6 w-6" />
                        <span className="text-sm font-medium">Dark</span>
                      </Label>
                    </div>
                    <div>
                      <RadioGroupItem value="system" id="system" className="peer sr-only" />
                      <Label
                        htmlFor="system"
                        className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary cursor-pointer"
                      >
                        <Monitor className="mb-3 h-6 w-6" />
                        <span className="text-sm font-medium">System</span>
                      </Label>
                    </div>
                  </RadioGroup>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  )
}

export default function SettingsPage() {
  return (
    <Suspense fallback={<SettingsPageSkeleton />}>
      <SettingsPageContent />
    </Suspense>
  )
}

function SettingsPageSkeleton() {
  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur">
        <div className="flex h-14 items-center justify-center gap-4 px-4 md:px-6">
          <div className="h-9 w-9 rounded-lg bg-muted animate-pulse" />
          <div className="space-y-1">
            <div className="h-5 w-20 bg-muted rounded animate-pulse" />
            <div className="h-3 w-40 bg-muted rounded animate-pulse hidden sm:block" />
          </div>
        </div>
      </header>
      <main className="flex justify-center w-full py-8 px-4 md:px-6">
        <div className="w-full max-w-4xl space-y-6">
          <div className="h-10 w-full bg-muted rounded animate-pulse" />
          <div className="h-64 w-full bg-muted rounded animate-pulse" />
        </div>
      </main>
    </div>
  )
}
