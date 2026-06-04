"use client"

import { useEffect, useMemo, useState } from "react"
import { useAppStore } from "@/lib/store"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { Cpu, Cloud, Info, RefreshCw, Loader2, Check, Trash2, Server, Plus } from "lucide-react"

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "/api/v1"

type OllamaModel = {
  name?: string
  model?: string
  size?: number
  modified_at?: string
  model_type?: "chat" | "embedding"
}

type ActiveModels = {
  chat_model: string | null
  embedding_model: string | null
  embedding_dimension: number | null
}

function formatBytes(value?: number) {
  if (!value) return "Unknown"
  const units = ["B", "KB", "MB", "GB", "TB"]
  let size = value
  let unitIndex = 0
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex += 1
  }
  return `${size.toFixed(1)} ${units[unitIndex]}`
}

export function ModelSourceSection() {
  const {
    useHostedModels,
    setUseHostedModels,
    setSelectedChatModel,
    setSelectedEmbeddingModel,
  } = useAppStore()
  const [models, setModels] = useState<OllamaModel[]>([])
  const [activeModels, setActiveModels] = useState<ActiveModels | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [actionInFlight, setActionInFlight] = useState<string | null>(null)
  const [pullModelName, setPullModelName] = useState("")
  const [pullModelType, setPullModelType] = useState<"chat" | "embedding">("chat")
  const [isPulling, setIsPulling] = useState(false)
  const [pullProgress, setPullProgress] = useState(0)
  const [filterType, setFilterType] = useState<"all" | "chat" | "embedding">("all")
  const [filterActiveOnly, setFilterActiveOnly] = useState(false)
  const [filterQuery, setFilterQuery] = useState("")

  const normalizedModels = useMemo(() => {
    return models
      .map((model) => ({
        name: model.name ?? model.model ?? "",
        size: model.size,
        modified_at: model.modified_at,
        model_type: model.model_type,
      }))
      .filter((model) => model.name)
  }, [models])

  const filteredModels = useMemo(() => {
    return normalizedModels.filter((model) => {
      if (filterType !== "all" && model.model_type !== filterType) {
        return false
      }
      if (filterActiveOnly) {
        const isActiveChat = activeModels?.chat_model === model.name
        const isActiveEmbedding = activeModels?.embedding_model === model.name
        if (!isActiveChat && !isActiveEmbedding) {
          return false
        }
      }
      if (filterQuery.trim()) {
        return model.name.toLowerCase().includes(filterQuery.trim().toLowerCase())
      }
      return true
    })
  }, [normalizedModels, filterType, filterActiveOnly, filterQuery, activeModels])

  const loadModels = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE}/model/list`)
      if (!response.ok) {
        throw new Error("Failed to load models")
      }
      const data = await response.json()
      setModels(data.models ?? [])
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load models")
    } finally {
      setIsLoading(false)
    }
  }

  const loadActiveModels = async () => {
    try {
      const response = await fetch(`${API_BASE}/model/active-models`)
      if (!response.ok) {
        throw new Error("Failed to load active models")
      }
      const data = await response.json()
      setActiveModels(data)
      setSelectedChatModel(data.chat_model ?? null)
      setSelectedEmbeddingModel(data.embedding_model ?? null)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load active models")
    }
  }

  const refreshAll = async () => {
    setError(null)
    await Promise.all([loadModels(), loadActiveModels()])
  }

  const activateModel = async (modelName: string, modelType: "chat" | "embedding") => {
    setActionInFlight(`${modelName}:${modelType}`)
    setError(null)
    try {
      const response = await fetch(`${API_BASE}/model/activate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: modelName, model_type: modelType }),
      })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || "Activation failed")
      }
      await loadActiveModels()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Activation failed")
    } finally {
      setActionInFlight(null)
    }
  }

  const deleteModel = async (modelName: string) => {
    setActionInFlight(`${modelName}:delete`)
    setError(null)
    try {
      const response = await fetch(`${API_BASE}/model/delete?model_name=${encodeURIComponent(modelName)}`, {
        method: "DELETE",
      })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || "Delete failed")
      }
      await refreshAll()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed")
    } finally {
      setActionInFlight(null)
    }
  }

  const pullModel = async () => {
    if (!pullModelName.trim()) return
    setIsPulling(true)
    setPullProgress(5)
    setError(null)
    try {
      const response = await fetch(`${API_BASE}/model/pull-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_name: pullModelName.trim(),
          model_type: pullModelType,
        }),
      })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || "Pull failed")
      }
      if (!response.body) {
        throw new Error("Pull stream unavailable")
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() ?? ""

        for (const line of lines) {
          if (!line.trim()) continue
          try {
            const payload = JSON.parse(line)
            if (payload.completed && payload.total) {
              const percent = Math.min(100, Math.round((payload.completed / payload.total) * 100))
              setPullProgress(percent)
            }
            if (payload.status === "success") {
              setPullProgress(100)
            }
          } catch {
            continue
          }
        }
      }
      setPullModelName("")
      setPullProgress(100)
      await refreshAll()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Pull failed")
    } finally {
      setTimeout(() => setPullProgress(0), 800)
      setIsPulling(false)
    }
  }

  useEffect(() => {
    if (!useHostedModels) {
      refreshAll()
    }
  }, [useHostedModels])

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

        {!useHostedModels && (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Server className="h-4 w-4" />
              <h4 className="font-semibold text-sm">Model Management</h4>
            </div>

            <div className="space-y-3 rounded-lg border border-dashed p-3">
              <Label htmlFor="pull-model">Pull a Model</Label>
              <div className="flex flex-wrap gap-2">
                <Input
                  id="pull-model"
                  placeholder="e.g., llama3.1:8b"
                  value={pullModelName}
                  onChange={(event) => setPullModelName(event.target.value)}
                  className="min-w-[220px] flex-1"
                />
                <Select value={pullModelType} onValueChange={(value) => setPullModelType(value as "chat" | "embedding")}>
                  <SelectTrigger className="w-[140px]">
                    <SelectValue placeholder="Model type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="chat">Chat</SelectItem>
                    <SelectItem value="embedding">Embedding</SelectItem>
                  </SelectContent>
                </Select>
                <Button onClick={pullModel} disabled={!pullModelName.trim() || isPulling}>
                  {isPulling ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
                  <span className="ml-2">Pull</span>
                </Button>
              </div>
              {isPulling && (
                <div className="space-y-2">
                  <Progress value={pullProgress} />
                  <p className="text-xs text-muted-foreground">Downloading model… {Math.round(pullProgress)}%</p>
                </div>
              )}
              <p className="text-xs text-muted-foreground">
                Pull downloads the model to the backend so it can be activated for chat or embeddings.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <Button variant="outline" onClick={refreshAll} disabled={isLoading}>
                {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                <span className="ml-2">Refresh</span>
              </Button>
              {activeModels && (
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <Badge variant="secondary">Chat: {activeModels.chat_model ?? "None"}</Badge>
                  <Badge variant="secondary">Embedding: {activeModels.embedding_model ?? "None"}</Badge>
                  {activeModels.embedding_dimension && (
                    <Badge variant="secondary">Dim: {activeModels.embedding_dimension}</Badge>
                  )}
                </div>
              )}
            </div>

            {error && (
              <Alert>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {normalizedModels.length === 0 && !isLoading ? (
              <div className="text-sm text-muted-foreground p-4 border rounded-lg border-dashed text-center">
                No models found. Pull a model from the backend first.
              </div>
            ) : (
              <div className="overflow-hidden rounded-lg border">
                <div className="flex flex-wrap items-center gap-2 border-b bg-muted/30 px-3 py-2">
                  <div className="flex items-center gap-2">
                    <Label className="text-xs text-muted-foreground">Filter</Label>
                    <Select value={filterType} onValueChange={(value) => setFilterType(value as typeof filterType)}>
                      <SelectTrigger className="h-8 w-[140px]">
                        <SelectValue placeholder="All types" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All</SelectItem>
                        <SelectItem value="chat">Chat</SelectItem>
                        <SelectItem value="embedding">Embedding</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <Button
                    size="sm"
                    variant={filterActiveOnly ? "default" : "outline"}
                    onClick={() => setFilterActiveOnly((prev) => !prev)}
                  >
                    Active Only
                  </Button>
                  <div className="ml-auto w-full sm:w-[220px]">
                    <Input
                      value={filterQuery}
                      onChange={(event) => setFilterQuery(event.target.value)}
                      placeholder="Search models"
                      className="h-8"
                    />
                  </div>
                </div>
                <div className="grid grid-cols-12 gap-2 border-b bg-muted/40 px-3 py-2 text-xs font-medium text-muted-foreground">
                  <div className="col-span-5">Model</div>
                  <div className="col-span-2">Size</div>
                  <div className="col-span-2">Updated</div>
                  <div className="col-span-2">Role</div>
                  <div className="col-span-1 text-right">Delete</div>
                </div>
                {filteredModels.length === 0 ? (
                  <div className="px-3 py-6 text-sm text-muted-foreground">
                    No models match the current filters.
                  </div>
                ) : (
                  filteredModels.map((model) => {
                    const isActiveChat = activeModels?.chat_model === model.name
                    const isActiveEmbedding = activeModels?.embedding_model === model.name
                    const isEmbeddingModel = model.model_type === "embedding"
                    return (
                      <div
                        key={model.name}
                        className="grid grid-cols-12 items-center gap-2 border-b px-3 py-3 text-sm last:border-b-0"
                      >
                        <div className="col-span-5 min-w-0">
                          <div className="font-medium truncate">{model.name}</div>
                          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                            {isActiveChat && <Badge className="text-xs">Active Chat</Badge>}
                            {isActiveEmbedding && <Badge className="text-xs">Active Embedding</Badge>}
                          </div>
                        </div>
                        <div className="col-span-2 text-xs text-muted-foreground">
                          {formatBytes(model.size)}
                        </div>
                        <div className="col-span-2 text-xs text-muted-foreground">
                          {model.modified_at ? new Date(model.modified_at).toLocaleDateString() : "-"}
                        </div>
                        <div className="col-span-2">
                          {isEmbeddingModel ? (
                            isActiveEmbedding ? (
                              <Badge className="text-xs">Embedding</Badge>
                            ) : (
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => activateModel(model.name, "embedding")}
                                disabled={actionInFlight !== null}
                              >
                                Set Embedding
                              </Button>
                            )
                          ) : isActiveChat ? (
                            <Badge className="text-xs">Chat</Badge>
                          ) : (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => activateModel(model.name, "chat")}
                              disabled={actionInFlight !== null}
                            >
                              Set Chat
                            </Button>
                          )}
                        </div>
                        <div className="col-span-1 flex justify-end">
                          <Button
                            size="icon"
                            variant="ghost"
                            className="text-muted-foreground hover:text-destructive"
                            onClick={() => deleteModel(model.name)}
                            disabled={actionInFlight !== null}
                          >
                            {actionInFlight === `${model.name}:delete` ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <Trash2 className="h-4 w-4" />
                            )}
                          </Button>
                        </div>
                      </div>
                    )
                  }))}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
