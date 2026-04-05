"use client"

import { useState, useCallback, useRef, useEffect } from "react"
import { useAppStore, type ContextSource } from "@/lib/store"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Textarea } from "@/components/ui/textarea"
import { Progress } from "@/components/ui/progress"
import {
  Link,
  Upload,
  FileText,
  Trash2,
  Plus,
  CheckCircle2,
  Loader2,
  AlertCircle,
  ExternalLink,
  Info,
  BookOpen,
  Files,
  X,
  Database,
  RefreshCw
} from "lucide-react"

export function ContextSection() {
  const {
    contextSources,
    indexingState,
    selectedEmbeddingModel,
    addContextSource,
    removeContextSource,
    updateContextSource,
    startIndexing,
    completeIndexing,
    setNeedsReindex,
  } = useAppStore()

  const [urlInput, setUrlInput] = useState("")
  const [isDragging, setIsDragging] = useState(false)
  const [indexingProgress, setIndexingProgress] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Track previous source count to detect new additions
  const prevSourceCountRef = useRef(contextSources.length)

  useEffect(() => {
    // If new sources were added, mark as needing reindex
    if (contextSources.length > prevSourceCountRef.current && indexingState.lastIndexedAt) {
      setNeedsReindex(true)
    }
    prevSourceCountRef.current = contextSources.length
  }, [contextSources.length, indexingState.lastIndexedAt, setNeedsReindex])

  const handleStartIndexing = () => {
    startIndexing()
    setIndexingProgress(0)

    // Simulate indexing progress
    const interval = setInterval(() => {
      setIndexingProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval)
          completeIndexing()
          return 100
        }
        return prev + Math.random() * 15
      })
    }, 300)
  }

  const handleAddUrls = () => {
    if (!urlInput.trim()) return

    // Split by newlines, commas, or spaces and filter empty entries
    const urls = urlInput
      .split(/[\n,\s]+/)
      .map(url => url.trim())
      .filter(url => url.length > 0)

    urls.forEach((url) => {
      // Add https:// if missing
      let fullUrl = url
      if (!url.startsWith('http://') && !url.startsWith('https://')) {
        fullUrl = `https://${url}`
      }

      let hostname = ""
      let displayName = url
      try {
        const urlObj = new URL(fullUrl)
        hostname = urlObj.hostname
        // Try to extract a meaningful name from the path
        const pathParts = urlObj.pathname.split('/').filter(Boolean)
        if (pathParts.length > 0) {
          displayName = pathParts[pathParts.length - 1].replace(/-/g, ' ')
          if (displayName.length > 40) {
            displayName = displayName.slice(0, 40) + '...'
          }
        } else {
          displayName = hostname
        }
      } catch {
        hostname = "Invalid URL"
        displayName = url.slice(0, 40)
      }

      const newSource: ContextSource = {
        id: crypto.randomUUID(),
        type: "url",
        name: displayName,
        url: fullUrl,
        status: "processing",
      }

      addContextSource(newSource)

      // Simulate processing with varied times
      setTimeout(() => {
        updateContextSource(newSource.id, { status: "ready" })
      }, 1500 + Math.random() * 2000)
    })

    setUrlInput("")
  }

  const handleFileUpload = useCallback((files: FileList | null) => {
    if (!files) return

    const pdfFiles = Array.from(files).filter(
      file => file.type === "application/pdf" || file.name.toLowerCase().endsWith('.pdf')
    )

    pdfFiles.forEach((file) => {
      const newSource: ContextSource = {
        id: crypto.randomUUID(),
        type: "pdf",
        name: file.name,
        status: "processing",
      }

      addContextSource(newSource)

      // Simulate processing with varied times based on file size
      const processingTime = Math.min(2000 + (file.size / 100000) * 500, 8000)
      setTimeout(() => {
        updateContextSource(newSource.id, { status: "ready" })
      }, processingTime)
    })
  }, [addContextSource, updateContextSource])

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    handleFileUpload(files)
    // Reset input
    e.target.value = ""
  }

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const files = e.dataTransfer.files
    handleFileUpload(files)
  }, [handleFileUpload])

  const getStatusBadge = (status: ContextSource["status"]) => {
    switch (status) {
      case "pending":
        return <Badge variant="secondary">Pending</Badge>
      case "processing":
        return (
          <Badge variant="secondary" className="gap-1">
            <Loader2 className="h-3 w-3 animate-spin" />
            Processing
          </Badge>
        )
      case "ready":
        return (
          <Badge variant="default" className="gap-1 bg-accent text-accent-foreground">
            <CheckCircle2 className="h-3 w-3" />
            Ready
          </Badge>
        )
      case "error":
        return (
          <Badge variant="destructive" className="gap-1">
            <AlertCircle className="h-3 w-3" />
            Error
          </Badge>
        )
    }
  }

  const urlSources = contextSources.filter(s => s.type === "url")
  const pdfSources = contextSources.filter(s => s.type === "pdf")

  const clearAllUrls = () => {
    urlSources.forEach(s => removeContextSource(s.id))
  }

  const clearAllPdfs = () => {
    pdfSources.forEach(s => removeContextSource(s.id))
  }

  return (
    <div className="space-y-6">
      {/* Indexing Card - Moved to top for visibility */}
      <Card className={`${indexingState.needsReindex ? "border-amber-500/50 ring-2 ring-amber-500/20" : contextSources.length > 0 && !indexingState.lastIndexedAt ? "border-primary/50 ring-2 ring-primary/20" : ""}`}>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Document Indexing
            {indexingState.needsReindex && (
              <Badge variant="secondary" className="bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30">
                Action Required
              </Badge>
            )}
          </CardTitle>
          <CardDescription>
            Index your documents to enable context-aware conversations. Re-indexing is required when you change the embedding model or add new sources.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className={`rounded-lg border p-4 ${indexingState.needsReindex
              ? "border-amber-500/50 bg-amber-500/5"
              : indexingState.lastIndexedAt
                ? "border-accent/50 bg-accent/5"
                : "border-border"
            }`}>
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  {indexingState.isIndexing ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin text-primary" />
                      <span className="font-medium">Indexing in progress...</span>
                    </>
                  ) : indexingState.needsReindex ? (
                    <>
                      <RefreshCw className="h-4 w-4 text-amber-500" />
                      <span className="font-medium text-amber-600 dark:text-amber-400">Re-indexing required</span>
                    </>
                  ) : indexingState.lastIndexedAt ? (
                    <>
                      <CheckCircle2 className="h-4 w-4 text-accent" />
                      <span className="font-medium">Index up to date</span>
                    </>
                  ) : contextSources.length > 0 ? (
                    <>
                      <AlertCircle className="h-4 w-4 text-primary" />
                      <span className="font-medium text-primary">Ready to index</span>
                    </>
                  ) : (
                    <>
                      <Info className="h-4 w-4 text-muted-foreground" />
                      <span className="font-medium text-muted-foreground">No documents added yet</span>
                    </>
                  )}
                </div>
                <div className="text-sm text-muted-foreground">
                  {indexingState.isIndexing ? (
                    <span>Processing {contextSources.length} source{contextSources.length !== 1 ? 's' : ''}...</span>
                  ) : indexingState.lastIndexedAt ? (
                    <span>
                      Last indexed: {new Date(indexingState.lastIndexedAt).toLocaleString()}
                      {indexingState.lastEmbeddingModel && (
                        <span className="ml-2 font-mono text-xs">({indexingState.lastEmbeddingModel})</span>
                      )}
                    </span>
                  ) : contextSources.length > 0 ? (
                    <span>{contextSources.length} source{contextSources.length !== 1 ? 's' : ''} ready for indexing</span>
                  ) : (
                    <span>Add Nature URLs or upload PDFs below to get started</span>
                  )}
                </div>
              </div>

              <Button
                onClick={handleStartIndexing}
                disabled={indexingState.isIndexing || contextSources.length === 0 || !selectedEmbeddingModel}
                size="lg"
                className={`min-w-[160px] ${indexingState.needsReindex ? "bg-amber-600 hover:bg-amber-700" : ""}`}
              >
                {indexingState.isIndexing ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Indexing...
                  </>
                ) : indexingState.needsReindex ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Re-Index Now
                  </>
                ) : (
                  <>
                    <Database className="h-4 w-4 mr-2" />
                    Start Indexing
                  </>
                )}
              </Button>
            </div>

            {indexingState.isIndexing && (
              <div className="mt-4 space-y-2">
                <Progress value={indexingProgress} className="h-2" />
                <p className="text-xs text-muted-foreground text-center">
                  {Math.round(indexingProgress)}% complete
                </p>
              </div>
            )}
          </div>

          {!selectedEmbeddingModel && contextSources.length > 0 && (
            <div className="flex items-start gap-2 rounded-lg bg-amber-500/10 border border-amber-500/30 p-3 text-sm">
              <AlertCircle className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
              <p className="text-amber-700 dark:text-amber-300">
                Please select an embedding model in the <span className="font-medium">Models</span> tab before indexing your documents.
              </p>
            </div>
          )}

          {/* Quick Stats */}
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-lg border bg-muted/30 p-3 text-center">
              <div className="text-2xl font-bold text-primary">{urlSources.length}</div>
              <p className="text-xs text-muted-foreground">URLs</p>
            </div>
            <div className="rounded-lg border bg-muted/30 p-3 text-center">
              <div className="text-2xl font-bold text-primary">{pdfSources.length}</div>
              <p className="text-xs text-muted-foreground">PDFs</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Nature Journal Access */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="h-5 w-5" />
            Nature Journal Access
          </CardTitle>
          <CardDescription>
            Connect to Nature journals for accessing research papers. Enter one or multiple URLs
            (separated by new lines or commas) to include them as context for your conversations.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="nature-urls">Paper or Collection URLs</Label>
              <span className="text-xs text-muted-foreground">
                {urlSources.length} URL{urlSources.length !== 1 ? 's' : ''} added
              </span>
            </div>
            <Textarea
              id="nature-urls"
              placeholder={`Enter one or more URLs, one per line:
https://www.nature.com/articles/s41586-024-...
https://www.nature.com/articles/s41591-024-...
nature.com/collections/...`}
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              className="min-h-[120px] font-mono text-sm"
            />
            <div className="flex gap-2">
              <Button onClick={handleAddUrls} disabled={!urlInput.trim()} className="flex-1">
                <Plus className="h-4 w-4 mr-1" />
                Add URLs
              </Button>
              {urlSources.length > 0 && (
                <Button variant="outline" onClick={clearAllUrls}>
                  <X className="h-4 w-4 mr-1" />
                  Clear All
                </Button>
              )}
            </div>
          </div>

          {/* URL list */}
          {urlSources.length > 0 && (
            <div className="space-y-2">
              <Label>Added URLs</Label>
              <div className="max-h-[200px] overflow-y-auto space-y-2 rounded-lg border p-2">
                {urlSources.map((source) => (
                  <div
                    key={source.id}
                    className="flex items-center justify-between p-2 rounded-md bg-muted/50"
                  >
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                      <Link className="h-4 w-4 shrink-0 text-primary" />
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium truncate">{source.name}</p>
                        {source.url && (
                          <p className="text-xs text-muted-foreground truncate">{source.url}</p>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      {getStatusBadge(source.status)}
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-muted-foreground hover:text-destructive"
                        onClick={() => removeContextSource(source.id)}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                        <span className="sr-only">Remove URL</span>
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex items-start gap-2 rounded-lg bg-muted/50 p-3 text-sm">
            <Info className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
            <div className="text-muted-foreground">
              <p>
                Enter full URLs of Nature papers or collections. The system will fetch and index
                the content for use in your conversations. Multiple URLs can be added at once.
              </p>
              <a
                href="https://www.nature.com"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-primary hover:underline mt-1"
              >
                Browse Nature journals <ExternalLink className="h-3 w-3" />
              </a>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* PDF Upload */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Files className="h-5 w-5" />
            PDF Documents
          </CardTitle>
          <CardDescription>
            Upload multiple PDF documents at once to use as context for your conversations.
            Documents are processed locally and their content is indexed for retrieval.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${isDragging
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/50"
              }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,application/pdf"
              multiple
              onChange={handleInputChange}
              className="hidden"
              id="pdf-upload"
            />
            <div className="flex flex-col items-center gap-3">
              <div className={`flex h-14 w-14 items-center justify-center rounded-full transition-colors ${isDragging ? "bg-primary/10" : "bg-muted"
                }`}>
                <Upload className={`h-7 w-7 ${isDragging ? "text-primary" : "text-muted-foreground"}`} />
              </div>
              <div>
                <p className="font-medium">
                  {isDragging ? "Drop PDFs here" : "Click to upload or drag and drop"}
                </p>
                <p className="text-sm text-muted-foreground">
                  Select multiple PDF files at once
                </p>
              </div>
              <p className="text-xs text-muted-foreground">
                Supports bulk upload. Maximum 50MB per file.
              </p>
            </div>
          </div>

          {/* PDF list */}
          {pdfSources.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Uploaded PDFs ({pdfSources.length})</Label>
                <Button variant="ghost" size="sm" onClick={clearAllPdfs} className="h-7 text-xs">
                  <X className="h-3 w-3 mr-1" />
                  Clear All
                </Button>
              </div>
              <div className="max-h-[250px] overflow-y-auto space-y-2 rounded-lg border p-2">
                {pdfSources.map((source) => (
                  <div
                    key={source.id}
                    className="flex items-center justify-between p-2 rounded-md bg-muted/50"
                  >
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                      <FileText className="h-4 w-4 shrink-0 text-destructive" />
                      <p className="text-sm font-medium truncate">{source.name}</p>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      {getStatusBadge(source.status)}
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-muted-foreground hover:text-destructive"
                        onClick={() => removeContextSource(source.id)}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                        <span className="sr-only">Remove PDF</span>
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex items-start gap-2 rounded-lg bg-muted/50 p-3 text-sm">
            <Info className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
            <p className="text-muted-foreground">
              PDFs are processed using OCR and text extraction. The content is indexed locally
              and used to provide relevant context when you ask questions. Select multiple files
              from your file browser or drag and drop a batch of PDFs.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Indexing & Summary Card */}
      <Card className={indexingState.needsReindex ? "border-amber-500/50" : ""}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Indexing & Summary
          </CardTitle>
          <CardDescription>
            Index your context sources for efficient retrieval during conversations.
            Re-indexing is needed when you change the embedding model or add new documents.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Indexing Status */}
          <div className={`rounded-lg border p-4 ${indexingState.needsReindex
              ? "border-amber-500/50 bg-amber-500/5"
              : indexingState.lastIndexedAt
                ? "border-accent/50 bg-accent/5"
                : "border-border"
            }`}>
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  {indexingState.isIndexing ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin text-primary" />
                      <span className="font-medium">Indexing in progress...</span>
                    </>
                  ) : indexingState.needsReindex ? (
                    <>
                      <RefreshCw className="h-4 w-4 text-amber-500" />
                      <span className="font-medium text-amber-600 dark:text-amber-400">Re-indexing required</span>
                    </>
                  ) : indexingState.lastIndexedAt ? (
                    <>
                      <CheckCircle2 className="h-4 w-4 text-accent" />
                      <span className="font-medium">Index up to date</span>
                    </>
                  ) : (
                    <>
                      <AlertCircle className="h-4 w-4 text-muted-foreground" />
                      <span className="font-medium">Not indexed yet</span>
                    </>
                  )}
                </div>
                <div className="text-sm text-muted-foreground">
                  {indexingState.isIndexing ? (
                    <span>Processing {contextSources.length} source{contextSources.length !== 1 ? 's' : ''}...</span>
                  ) : indexingState.lastIndexedAt ? (
                    <span>
                      Last indexed: {new Date(indexingState.lastIndexedAt).toLocaleString()}
                      {indexingState.lastEmbeddingModel && (
                        <span className="ml-2 font-mono text-xs">({indexingState.lastEmbeddingModel})</span>
                      )}
                    </span>
                  ) : (
                    <span>Click &quot;Start Indexing&quot; to enable context-aware conversations</span>
                  )}
                </div>
              </div>

              <Button
                onClick={handleStartIndexing}
                disabled={indexingState.isIndexing || contextSources.length === 0 || !selectedEmbeddingModel}
                className={indexingState.needsReindex ? "bg-amber-600 hover:bg-amber-700" : ""}
              >
                {indexingState.isIndexing ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Indexing...
                  </>
                ) : indexingState.needsReindex ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Re-Index Now
                  </>
                ) : (
                  <>
                    <Database className="h-4 w-4 mr-2" />
                    Start Indexing
                  </>
                )}
              </Button>
            </div>

            {indexingState.isIndexing && (
              <div className="mt-4 space-y-2">
                <Progress value={indexingProgress} className="h-2" />
                <p className="text-xs text-muted-foreground text-center">
                  {Math.round(indexingProgress)}% complete
                </p>
              </div>
            )}
          </div>

          {!selectedEmbeddingModel && contextSources.length > 0 && (
            <div className="flex items-start gap-2 rounded-lg bg-amber-500/10 border border-amber-500/30 p-3 text-sm">
              <AlertCircle className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
              <p className="text-amber-700 dark:text-amber-300">
                Please select an embedding model in the Models tab before indexing your documents.
              </p>
            </div>
          )}

          {/* Stats Grid */}
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="rounded-lg border p-4 text-center">
              <div className="text-3xl font-bold text-primary">{contextSources.length}</div>
              <p className="text-sm text-muted-foreground">Total Sources</p>
            </div>
            <div className="rounded-lg border p-4 text-center">
              <div className="text-3xl font-bold text-primary">{urlSources.length}</div>
              <p className="text-sm text-muted-foreground">URLs Added</p>
            </div>
            <div className="rounded-lg border p-4 text-center">
              <div className="text-3xl font-bold text-primary">{pdfSources.length}</div>
              <p className="text-sm text-muted-foreground">PDFs Uploaded</p>
            </div>
          </div>

          {contextSources.length > 0 && (
            <div className="flex items-center justify-between rounded-lg bg-muted/50 p-3">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-accent" />
                <span className="text-sm">
                  {contextSources.filter(s => s.status === "ready").length} of {contextSources.length} sources ready
                </span>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => contextSources.forEach(s => removeContextSource(s.id))}
              >
                Clear All Sources
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
