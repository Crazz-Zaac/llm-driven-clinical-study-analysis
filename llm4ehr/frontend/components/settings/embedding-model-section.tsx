"use client"

import { useAppStore } from "@/lib/store"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Database, Download, Trash2, Check, Loader2, Plus, Info } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { useState } from "react"
import { Input } from "@/components/ui/input"

export function EmbeddingModelSection() {
    const {
        localModels,
        selectedEmbeddingModel,
        useHostedModels,
        setSelectedEmbeddingModel,
        addLocalModel,
        updateLocalModel,
        removeLocalModel,
        indexingState,
        setNeedsReindex,
    } = useAppStore()

    const [newModelName, setNewModelName] = useState("")
    const [isDownloading, setIsDownloading] = useState<string | null>(null)

    const embeddingModels = localModels.filter(m => m.type === "embedding")
    const downloadedEmbeddings = embeddingModels.filter(m => m.status === "downloaded")

    const handleEmbeddingModelChange = (modelId: string) => {
        setSelectedEmbeddingModel(modelId)
        // Trigger reindex if there are context sources and the model changed
        if (indexingState.lastEmbeddingModel && indexingState.lastEmbeddingModel !== modelId) {
            setNeedsReindex(true)
        }
    }

    const handleAddModel = () => {
        if (!newModelName.trim()) return

        const modelId = newModelName.toLowerCase().replace(/[^a-z0-9]/g, "-")

        // Check if model already exists
        if (localModels.some(m => m.id === modelId)) {
            return
        }

        addLocalModel({
            id: modelId,
            name: newModelName.trim(),
            size: "Calculating...",
            type: "embedding",
            status: "downloading",
            downloadProgress: 0,
        })

        setIsDownloading(modelId)
        setNewModelName("")

        // Simulate download progress
        let progress = 0
        const interval = setInterval(() => {
            progress += Math.random() * 15
            if (progress >= 100) {
                updateLocalModel(modelId, {
                    status: "downloaded",
                    downloadProgress: 100,
                    size: `${(Math.random() * 2 + 0.3).toFixed(1)} GB`
                })
                setIsDownloading(null)
                clearInterval(interval)
            } else {
                updateLocalModel(modelId, { downloadProgress: Math.min(progress, 99) })
            }
        }, 300)
    }

    const handleDownload = (id: string) => {
        setIsDownloading(id)
        updateLocalModel(id, { status: "downloading", downloadProgress: 0 })

        let progress = 0
        const interval = setInterval(() => {
            progress += Math.random() * 15
            if (progress >= 100) {
                updateLocalModel(id, { status: "downloaded", downloadProgress: 100 })
                setIsDownloading(null)
                clearInterval(interval)
            } else {
                updateLocalModel(id, { downloadProgress: Math.min(progress, 99) })
            }
        }, 300)
    }

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Database className="h-5 w-5" />
                    Embedding Model
                </CardTitle>
                <CardDescription>
                    {useHostedModels
                        ? "Select an embedding model to generate vector representations of your documents for semantic search. Required for context awareness across both local and hosted chat models."
                        : "Download and manage embedding models used to generate vector representations of your documents for semantic search. Required for context awareness."}
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
                {/* Add New Model (Local only) */}
                {!useHostedModels && (
                    <div className="space-y-3">
                        <Label>Add New Embedding Model</Label>
                        <div className="flex gap-2">
                            <Input
                                placeholder="Enter model name (e.g., all-MiniLM-L6-v2)"
                                value={newModelName}
                                onChange={(e) => setNewModelName(e.target.value)}
                                onKeyDown={(e) => e.key === "Enter" && handleAddModel()}
                                className="flex-1"
                            />
                            <Button onClick={handleAddModel} disabled={!newModelName.trim() || isDownloading !== null}>
                                <Plus className="h-4 w-4 mr-1" />
                                Add
                            </Button>
                        </div>
                        <p className="text-xs text-muted-foreground">
                            Enter a model identifier. For local models, use Hugging Face model names like 'sentence-transformers/all-MiniLM-L6-v2'.
                        </p>
                    </div>
                )}

                {/* Model Selection Dropdown */}
                {downloadedEmbeddings.length > 0 && (
                    <div className="space-y-3">
                        <Label htmlFor="embedding-select">Select Embedding Model</Label>
                        <Select
                            value={selectedEmbeddingModel || ""}
                            onValueChange={handleEmbeddingModelChange}
                        >
                            <SelectTrigger id="embedding-select">
                                <SelectValue placeholder="Choose an embedding model" />
                            </SelectTrigger>
                            <SelectContent>
                                {downloadedEmbeddings.map((model) => (
                                    <SelectItem key={model.id} value={model.id}>
                                        {model.name}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>
                )}

                {/* Model List (Local only) */}
                {!useHostedModels && (
                    <div className="space-y-3">
                        <Label>Available Models</Label>
                        <div className="space-y-2">
                            {embeddingModels.length === 0 ? (
                                <div className="text-sm text-muted-foreground p-4 border rounded-lg border-dashed text-center">
                                    No embedding models added. Add a model above to get started.
                                </div>
                            ) : (
                                embeddingModels.map((model) => (
                                    <EmbeddingModelRow
                                        key={model.id}
                                        model={model}
                                        isSelected={selectedEmbeddingModel === model.id}
                                        onSelect={() => handleEmbeddingModelChange(model.id)}
                                        onDownload={() => handleDownload(model.id)}
                                        onRemove={() => removeLocalModel(model.id)}
                                    />
                                ))
                            )}
                        </div>
                    </div>
                )}

                <div className="flex items-start gap-2 rounded-lg bg-muted/50 p-3 text-sm">
                    <Info className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
                    <p className="text-muted-foreground">
                        The embedding model is used to create vector representations of documents. It must be selected
                        for context-aware search and retrieval to work properly with both local and hosted chat models.
                    </p>
                </div>
            </CardContent>
        </Card>
    )
}

interface EmbeddingModelRowProps {
    model: {
        id: string
        name: string
        size: string
        status: "available" | "downloading" | "downloaded"
        downloadProgress?: number
    }
    isSelected: boolean
    onSelect: () => void
    onDownload: () => void
    onRemove: () => void
}

function EmbeddingModelRow({ model, isSelected, onSelect, onDownload, onRemove }: EmbeddingModelRowProps) {
    return (
        <div className={`flex items-center justify-between p-3 rounded-lg border transition-colors ${isSelected ? "border-primary bg-primary/5" : "border-border"
            }`}>
            <div className="flex items-center gap-3 min-w-0">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-muted">
                    <Database className="h-4 w-4" />
                </div>
                <div className="min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-medium truncate">{model.name}</span>
                        <Badge variant="secondary" className="text-xs">{model.size}</Badge>
                        {isSelected && <Badge variant="default" className="text-xs">Active</Badge>}
                    </div>
                    <p className="text-xs text-muted-foreground">{model.id}</p>
                </div>
            </div>

            <div className="flex items-center gap-2 shrink-0">
                {model.status === "available" && (
                    <Button size="sm" variant="outline" onClick={onDownload}>
                        <Download className="h-4 w-4 mr-1" />
                        Download
                    </Button>
                )}
                {model.status === "downloading" && (
                    <div className="flex items-center gap-2">
                        <Progress value={model.downloadProgress} className="w-20 h-2" />
                        <span className="text-xs text-muted-foreground w-8">{Math.round(model.downloadProgress || 0)}%</span>
                        <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                    </div>
                )}
                {model.status === "downloaded" && (
                    <>
                        <Button
                            size="sm"
                            variant={isSelected ? "default" : "outline"}
                            onClick={onSelect}
                        >
                            {isSelected ? (
                                <>
                                    <Check className="h-4 w-4 mr-1" />
                                    Selected
                                </>
                            ) : (
                                "Select"
                            )}
                        </Button>
                        <Button
                            size="sm"
                            variant="ghost"
                            onClick={onRemove}
                            className="text-muted-foreground hover:text-destructive"
                        >
                            <Trash2 className="h-4 w-4" />
                        </Button>
                    </>
                )}
            </div>
        </div>
    )
}
