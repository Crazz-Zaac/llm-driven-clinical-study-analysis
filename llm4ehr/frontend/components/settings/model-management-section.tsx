"use client"

import { useEffect, useMemo, useState } from "react"
import { useAppStore } from "@/lib/store"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2, RefreshCw, Check, Trash2, Server } from "lucide-react"

const API_BASE = "/api/v1"

type OllamaModel = {
    name?: string
    model?: string
    size?: number
    modified_at?: string
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

export function ModelManagementSection() {
    const { useHostedModels } = useAppStore()
    const [models, setModels] = useState<OllamaModel[]>([])
    const [activeModels, setActiveModels] = useState<ActiveModels | null>(null)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [actionInFlight, setActionInFlight] = useState<string | null>(null)

    const normalizedModels = useMemo(() => {
        return models
            .map((model) => ({
                name: model.name ?? model.model ?? "",
                size: model.size,
                modified_at: model.modified_at,
            }))
            .filter((model) => model.name)
    }, [models])

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
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load active models")
        }
    }

    const refreshAll = async () => {
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

    useEffect(() => {
        if (!useHostedModels) {
            refreshAll()
        }
    }, [useHostedModels])

    if (useHostedModels) {
        return null
    }

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Server className="h-5 w-5" />
                    Model Management
                </CardTitle>
                <CardDescription>
                    Manage Ollama models on the backend. Activate a model for chat or embeddings, or delete unused models.
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
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
                    <div className="space-y-3">
                        {normalizedModels.map((model) => {
                            const isActiveChat = activeModels?.chat_model === model.name
                            const isActiveEmbedding = activeModels?.embedding_model === model.name
                            return (
                                <div
                                    key={model.name}
                                    className="flex flex-wrap items-center justify-between gap-3 rounded-lg border p-3"
                                >
                                    <div className="min-w-0">
                                        <div className="flex items-center gap-2 flex-wrap">
                                            <span className="font-medium truncate">{model.name}</span>
                                            <Badge variant="secondary" className="text-xs">{formatBytes(model.size)}</Badge>
                                            {model.modified_at && (
                                                <Badge variant="outline" className="text-xs">{new Date(model.modified_at).toLocaleDateString()}</Badge>
                                            )}
                                            {isActiveChat && <Badge className="text-xs">Active Chat</Badge>}
                                            {isActiveEmbedding && <Badge className="text-xs">Active Embedding</Badge>}
                                        </div>
                                    </div>
                                    <div className="flex flex-wrap items-center gap-2">
                                        <Button
                                            size="sm"
                                            variant={isActiveChat ? "default" : "outline"}
                                            onClick={() => activateModel(model.name, "chat")}
                                            disabled={actionInFlight !== null}
                                        >
                                            {isActiveChat ? <Check className="h-4 w-4 mr-1" /> : null}
                                            Chat
                                        </Button>
                                        <Button
                                            size="sm"
                                            variant={isActiveEmbedding ? "default" : "outline"}
                                            onClick={() => activateModel(model.name, "embedding")}
                                            disabled={actionInFlight !== null}
                                        >
                                            {isActiveEmbedding ? <Check className="h-4 w-4 mr-1" /> : null}
                                            Embedding
                                        </Button>
                                        <Button
                                            size="sm"
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
                        })}
                    </div>
                )}
            </CardContent>
        </Card>
    )
}
